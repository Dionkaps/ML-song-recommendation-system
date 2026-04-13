from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List


WORKSPACE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKSPACE_DIR.parent
DEFAULT_SOURCE_SCRIPT = WORKSPACE_DIR / "msd_deezer_pipeline.py"
DEFAULT_SUBSET_DIR = PROJECT_ROOT / "millionsongsubset" / "MillionSongSubset"
DEFAULT_CSV_PATH = WORKSPACE_DIR / "data" / "msd_deezer_matches.csv"
DEFAULT_AUDIO_DIR = WORKSPACE_DIR / "audio"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the MSD Deezer pipeline in restartable sessions and terminate any "
            "session that goes idle for too long."
        )
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch the subprocess.")
    parser.add_argument("--source-script", default=str(DEFAULT_SOURCE_SCRIPT), help="Path to the pipeline script.")
    parser.add_argument("--subset-dir", default=str(DEFAULT_SUBSET_DIR), help="Path to the MillionSongSubset folder.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH), help="Path to the output CSV.")
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR), help="Directory containing downloaded audio files.")
    parser.add_argument("--chunk-size", type=int, default=500, help="How many songs each subprocess session should attempt.")
    parser.add_argument("--idle-timeout-sec", type=int, default=240, help="Kill and restart the subprocess if it produces no activity for this long.")
    parser.add_argument("--poll-interval-sec", type=float, default=5.0, help="Polling interval used while monitoring the subprocess.")
    parser.add_argument("--max-stall-restarts", type=int, default=20, help="Maximum number of forced restarts after idle stalls.")
    parser.add_argument("--max-no-progress-sessions", type=int, default=2, help="Stop after this many completed sessions with no new CSV/audio progress.")
    parser.add_argument("--max-sessions", type=int, default=0, help="Optional hard cap on how many subprocess sessions to run. Use 1 for a tiny smoke test.")
    parser.add_argument("--save-every", type=int, default=25, help="Persist the CSV and cache after this many attempts inside the pipeline.")
    parser.add_argument("--request-delay", type=float, default=0.0, help="Delay between Deezer attempts inside the pipeline (single-worker mode, 0 when using rate limiter).")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent search threads inside the pipeline (download threads = workers * 2).")
    parser.add_argument("--max-songs-per-sec", type=float, default=8.0, help="Initial rate-limit cap on API calls/sec; adapts automatically on 429s (0 = unlimited).")
    parser.add_argument("--retry-no-match", action="store_true", help="Pass through retry-no-match to the pipeline.")
    parser.add_argument("--redownload-existing", action="store_true", help="Pass through redownload-existing to the pipeline.")
    parser.add_argument("--log-dir", default="", help="Optional directory for per-session logs and the run summary.")
    return parser.parse_args()


def count_subset_rows(subset_dir: Path) -> int:
    if not subset_dir.exists():
        return 0
    return sum(1 for _ in subset_dir.rglob("*.h5"))


def classify_csv_row(row: dict[str, str]) -> str:
    match_status = str(row.get("deezer_match_status", "") or "").strip()
    download_status = str(row.get("deezer_download_status", "") or "").strip()

    if download_status == "downloaded":
        return "downloaded"
    if match_status == "no_match":
        return "no_match"
    if match_status or download_status:
        return "retryable"
    return "untouched"


def load_csv_progress(csv_path: Path) -> Dict[str, int | None]:
    if not csv_path.exists():
        return {
            "total": 0,
            "downloaded": 0,
            "no_match": 0,
            "retryable": 0,
            "attempted": 0,
            "untouched": 0,
            "first_untouched_index": 0,
            "first_unfinished_index": 0,
        }

    last_error = None
    for _ in range(3):
        try:
            total = 0
            downloaded = 0
            no_match = 0
            retryable = 0
            first_untouched_index = None
            first_unfinished_index = None

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for index, row in enumerate(reader):
                    state = classify_csv_row(row)
                    total += 1
                    if state == "downloaded":
                        downloaded += 1
                    elif state == "no_match":
                        no_match += 1
                    elif state == "retryable":
                        retryable += 1
                    else:
                        if first_untouched_index is None:
                            first_untouched_index = index

                    if state in {"retryable", "untouched"} and first_unfinished_index is None:
                        first_unfinished_index = index

            attempted = downloaded + no_match + retryable
            untouched = max(0, total - attempted)
            return {
                "total": total,
                "downloaded": downloaded,
                "no_match": no_match,
                "retryable": retryable,
                "attempted": attempted,
                "untouched": untouched,
                "first_untouched_index": first_untouched_index if first_untouched_index is not None else total,
                "first_unfinished_index": first_unfinished_index if first_unfinished_index is not None else total,
            }
        except Exception as exc:
            last_error = exc
            time.sleep(0.2)

    raise RuntimeError(f"Could not read CSV progress from {csv_path}: {last_error}")


def count_audio_files(audio_dir: Path) -> int:
    if not audio_dir.exists():
        return 0
    return sum(
        1
        for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a"}
    )


def _reader_thread(stream, queue: Queue[str]) -> None:
    try:
        while True:
            chunk = stream.readline()
            if not chunk:
                break
            queue.put(chunk)
    finally:
        stream.close()


def ensure_log_dir(log_dir_arg: str) -> Path:
    if log_dir_arg:
        path = Path(log_dir_arg)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = WORKSPACE_DIR / "logs" / f"resilient_download_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def terminate_process(proc: subprocess.Popen[str], grace_sec: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    deadline = time.time() + grace_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    if proc.poll() is None:
        proc.kill()


def write_console(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    data = str(text).encode(encoding, errors="replace")
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        buffer.write(data)
    else:
        sys.stdout.write(data.decode(encoding, errors="replace"))
    sys.stdout.flush()


def build_command(
    args: argparse.Namespace,
    source_script: Path,
    subset_dir: Path,
    csv_path: Path,
    audio_dir: Path,
    start_index: int,
    limit: int,
) -> List[str]:
    command = [
        str(args.python),
        "-u",
        str(source_script),
        "--subset-dir",
        str(subset_dir),
        "--csv-path",
        str(csv_path),
        "--audio-dir",
        str(audio_dir),
        "--start-index",
        str(max(0, start_index)),
        "--limit",
        str(max(1, limit)),
        "--save-every",
        str(max(1, int(args.save_every))),
        "--request-delay",
        str(max(0.0, float(args.request_delay))),
        "--workers",
        str(max(1, int(args.workers))),
        "--max-songs-per-sec",
        str(max(0.0, float(args.max_songs_per_sec))),
    ]

    if csv_path.exists():
        command.append("--skip-extract")
    if args.retry_no_match:
        command.append("--retry-no-match")
    if args.redownload_existing:
        command.append("--redownload-existing")

    return command


def choose_session_start(progress: Dict[str, int | None]) -> int:
    untouched = int(progress["untouched"])
    if untouched > 0:
        return int(progress["first_untouched_index"])
    return int(progress["first_unfinished_index"])


def main() -> None:
    args = parse_args()
    source_script = Path(args.source_script).resolve()
    subset_dir = Path(args.subset_dir).resolve()
    csv_path = Path(args.csv_path).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    log_dir = ensure_log_dir(args.log_dir)
    run_summary_path = log_dir / "run_summary.json"

    if not source_script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {source_script}")

    csv_progress = load_csv_progress(csv_path) if csv_path.exists() else {"total": 0}
    total_catalog_rows = int(csv_progress.get("total", 0) or 0)
    if total_catalog_rows <= 0:
        if not subset_dir.exists():
            raise FileNotFoundError(
                f"No CSV found at {csv_path} and subset directory not found at {subset_dir}. "
                "Either provide a pre-built CSV or a valid --subset-dir."
            )
        total_catalog_rows = count_subset_rows(subset_dir)

    forced_restart_count = 0
    no_progress_sessions = 0
    session_summaries: List[Dict[str, object]] = []

    print(
        json.dumps(
            {
                "catalog_rows": total_catalog_rows,
                "chunk_size": int(args.chunk_size),
                "idle_timeout_sec": int(args.idle_timeout_sec),
                "log_dir": str(log_dir),
            },
            indent=2,
        )
    )

    while True:
        csv_before = load_csv_progress(csv_path) if csv_path.exists() else {
            "total": total_catalog_rows,
            "downloaded": 0,
            "no_match": 0,
            "retryable": 0,
            "attempted": 0,
            "untouched": total_catalog_rows,
            "first_untouched_index": 0,
            "first_unfinished_index": 0,
        }
        audio_before = count_audio_files(audio_dir)
        session_csv_before = dict(csv_before)
        session_audio_before = int(audio_before)

        remaining_untouched = int(csv_before["untouched"])
        remaining_retryable = int(csv_before["retryable"])
        if remaining_untouched <= 0 and remaining_retryable <= 0:
            break

        remaining_target = remaining_untouched if remaining_untouched > 0 else remaining_retryable
        session_index = len(session_summaries) + 1
        session_limit = min(int(args.chunk_size), remaining_target)
        session_start_index = choose_session_start(csv_before)
        log_path = log_dir / f"{session_index:02d}_download_session.log"
        command = build_command(
            args=args,
            source_script=source_script,
            subset_dir=subset_dir,
            csv_path=csv_path,
            audio_dir=audio_dir,
            start_index=session_start_index,
            limit=session_limit,
        )

        print(
            f"\n[resilient-msd-deezer] session={session_index} start_index={session_start_index} "
            f"limit={session_limit} untouched={remaining_untouched} retryable={remaining_retryable}"
        )
        proc = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )

        assert proc.stdout is not None
        queue: Queue[str] = Queue()
        reader = threading.Thread(target=_reader_thread, args=(proc.stdout, queue), daemon=True)
        reader.start()

        session_start = time.time()
        last_activity = session_start
        stalled = False

        with log_path.open("w", encoding="utf-8") as log_handle:
            while True:
                drained = False
                while True:
                    try:
                        line = queue.get_nowait()
                    except Empty:
                        break
                    drained = True
                    last_activity = time.time()
                    write_console(line)
                    log_handle.write(line)
                    log_handle.flush()

                csv_now = load_csv_progress(csv_path) if csv_path.exists() else csv_before
                audio_now = count_audio_files(audio_dir)
                if (
                    int(csv_now["attempted"]) > int(csv_before["attempted"])
                    or int(csv_now["downloaded"]) > int(csv_before["downloaded"])
                    or audio_now > audio_before
                ):
                    last_activity = time.time()
                    csv_before = csv_now
                    audio_before = audio_now

                exit_code = proc.poll()
                if exit_code is not None:
                    while True:
                        try:
                            line = queue.get_nowait()
                        except Empty:
                            break
                        last_activity = time.time()
                        write_console(line)
                        log_handle.write(line)
                    log_handle.flush()
                    break

                idle_for = time.time() - last_activity
                if idle_for >= float(args.idle_timeout_sec):
                    stalled = True
                    forced_restart_count += 1
                    message = (
                        f"\n[resilient-msd-deezer] idle timeout after {idle_for:.1f}s; "
                        "terminating stalled pipeline session.\n"
                    )
                    write_console(message)
                    log_handle.write(message)
                    log_handle.flush()
                    terminate_process(proc)
                    break

                if not drained:
                    time.sleep(float(args.poll_interval_sec))

        reader.join(timeout=5.0)

        csv_after = load_csv_progress(csv_path) if csv_path.exists() else csv_before
        audio_after = count_audio_files(audio_dir)
        session_summary: Dict[str, object] = {
            "session_index": session_index,
            "started_at": datetime.fromtimestamp(session_start).isoformat(timespec="seconds"),
            "duration_sec": round(time.time() - session_start, 2),
            "stalled": stalled,
            "exit_code": proc.returncode,
            "start_index": session_start_index,
            "limit": session_limit,
            "attempted_before": int(session_csv_before["attempted"]),
            "downloaded_before": int(session_csv_before["downloaded"]),
            "audio_before": session_audio_before,
            "attempted_after": int(csv_after["attempted"]),
            "downloaded_after": int(csv_after["downloaded"]),
            "audio_after": int(audio_after),
            "log_path": str(log_path),
        }
        session_summary["attempted_gain"] = int(session_summary["attempted_after"]) - int(session_summary["attempted_before"])
        session_summary["downloaded_gain"] = int(session_summary["downloaded_after"]) - int(session_summary["downloaded_before"])
        session_summary["audio_gain"] = int(session_summary["audio_after"]) - int(session_summary["audio_before"])

        progress_made = any(
            int(session_summary[key]) > 0
            for key in ("attempted_gain", "downloaded_gain", "audio_gain")
        )
        session_summaries.append(session_summary)

        run_summary = {
            "catalog_rows": total_catalog_rows,
            "forced_restart_count": forced_restart_count,
            "sessions": session_summaries,
        }
        run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

        if stalled and forced_restart_count >= int(args.max_stall_restarts):
            raise RuntimeError(
                "Pipeline exceeded the allowed number of idle restarts. "
                f"See {run_summary_path} for details."
            )

        if progress_made:
            no_progress_sessions = 0
        else:
            no_progress_sessions += 1
            print(
                f"[resilient-msd-deezer] session {session_index} finished with no new progress "
                f"(count={no_progress_sessions})."
            )
            if no_progress_sessions >= int(args.max_no_progress_sessions):
                print(
                    "[resilient-msd-deezer] stopping after repeated no-progress sessions. "
                    "Remaining songs could not be advanced in this run."
                )
                break

        if int(args.max_sessions) > 0 and len(session_summaries) >= int(args.max_sessions):
            print(
                f"[resilient-msd-deezer] reached max session cap ({args.max_sessions}); "
                "stopping after the requested smoke test."
            )
            break

    final_csv = load_csv_progress(csv_path) if csv_path.exists() else {
        "total": total_catalog_rows,
        "downloaded": 0,
        "no_match": 0,
        "retryable": 0,
        "attempted": 0,
        "untouched": total_catalog_rows,
    }
    final_audio = count_audio_files(audio_dir)
    payload = {
        "catalog_rows": total_catalog_rows,
        "attempted_rows": int(final_csv["attempted"]),
        "downloaded_rows": int(final_csv["downloaded"]),
        "no_match_rows": int(final_csv["no_match"]),
        "retryable_rows": int(final_csv["retryable"]),
        "untouched_rows": int(final_csv["untouched"]),
        "audio_files": final_audio,
        "forced_restart_count": forced_restart_count,
        "sessions": session_summaries,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    run_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nResilient MSD Deezer run finished.")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
