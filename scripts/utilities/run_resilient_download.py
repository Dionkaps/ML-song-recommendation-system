from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CATALOG_CSV = PROJECT_ROOT / "data" / "millionsong_dataset.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "download_checkpoint_with_genre.json"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "audio_files"
DEFAULT_SOURCE_SCRIPT = PROJECT_ROOT / "src" / "data_collection" / "deezer-song.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Deezer preview downloader in restartable chunks and kill any "
            "session that goes idle for too long."
        )
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the downloader subprocess.",
    )
    parser.add_argument(
        "--source-script",
        default=str(DEFAULT_SOURCE_SCRIPT),
        help="Path to the downloader script.",
    )
    parser.add_argument(
        "--catalog-csv",
        default=str(DEFAULT_CATALOG_CSV),
        help="Path to the Million Song catalog CSV.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the downloader checkpoint JSON.",
    )
    parser.add_argument(
        "--audio-dir",
        default=str(DEFAULT_AUDIO_DIR),
        help="Directory containing downloaded audio files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="How many remaining songs each subprocess session should target.",
    )
    parser.add_argument(
        "--idle-timeout-sec",
        type=int,
        default=240,
        help="Kill and restart the subprocess if it produces no activity for this long.",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=float,
        default=5.0,
        help="Polling interval used while monitoring the subprocess.",
    )
    parser.add_argument(
        "--max-stall-restarts",
        type=int,
        default=20,
        help="Maximum number of forced restarts after idle stalls.",
    )
    parser.add_argument(
        "--max-no-progress-sessions",
        type=int,
        default=2,
        help="Stop after this many completed sessions with no new checkpoint/audio progress.",
    )
    parser.add_argument(
        "--log-dir",
        default="",
        help="Optional directory for per-session logs and the run summary.",
    )
    return parser.parse_args()


def count_catalog_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def load_checkpoint_counts(checkpoint_path: Path) -> Dict[str, int]:
    if not checkpoint_path.exists():
        return {"processed": 0, "downloaded": 0}
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return {"processed": 0, "downloaded": 0}

    processed_indices = set(payload.get("processed_indices", []))
    failed_indices = set(payload.get("failed_indices", []))
    downloaded_keys = {
        str(item.get("msd_track_id") or item.get("deezer_track_id") or item.get("filename") or "").strip()
        for item in payload.get("downloaded_songs", [])
        if str(item.get("msd_track_id") or item.get("deezer_track_id") or item.get("filename") or "").strip()
    }

    return {
        "processed": int(len(processed_indices | failed_indices)),
        "downloaded": int(len(downloaded_keys)),
    }


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
        path = PROJECT_ROOT / "docs" / "reports" / "run_logs" / f"resilient_download_{stamp}"
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


def main() -> None:
    args = parse_args()
    source_script = Path(args.source_script)
    catalog_csv = Path(args.catalog_csv)
    checkpoint_path = Path(args.checkpoint)
    audio_dir = Path(args.audio_dir)
    log_dir = ensure_log_dir(args.log_dir)
    run_summary_path = log_dir / "run_summary.json"

    if not catalog_csv.exists():
        raise FileNotFoundError(f"Catalog CSV not found: {catalog_csv}")
    if not source_script.exists():
        raise FileNotFoundError(f"Downloader script not found: {source_script}")

    total_catalog_rows = count_catalog_rows(catalog_csv)
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
        checkpoint_before = load_checkpoint_counts(checkpoint_path)
        audio_before = count_audio_files(audio_dir)
        session_checkpoint_before = dict(checkpoint_before)
        session_audio_before = int(audio_before)
        remaining = max(0, total_catalog_rows - session_checkpoint_before["processed"])
        if remaining <= 0:
            break

        session_index = len(session_summaries) + 1
        session_limit = min(int(args.chunk_size), remaining)
        log_path = log_dir / f"{session_index:02d}_download_session.log"
        command = [
            str(args.python),
            "-u",
            str(source_script),
            "--limit",
            str(session_limit),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        print(
            f"\n[resilient-download] session={session_index} remaining={remaining} "
            f"limit={session_limit}"
        )
        proc = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            encoding="utf-8",
            errors="replace",
        )

        assert proc.stdout is not None
        queue: Queue[str] = Queue()
        reader = threading.Thread(
            target=_reader_thread,
            args=(proc.stdout, queue),
            daemon=True,
        )
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

                checkpoint_now = load_checkpoint_counts(checkpoint_path)
                audio_now = count_audio_files(audio_dir)
                if (
                    checkpoint_now["processed"] > checkpoint_before["processed"]
                    or checkpoint_now["downloaded"] > checkpoint_before["downloaded"]
                    or audio_now > audio_before
                ):
                    last_activity = time.time()
                    checkpoint_before = checkpoint_now
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
                        f"\n[resilient-download] idle timeout after {idle_for:.1f}s; "
                        "terminating stalled downloader session.\n"
                    )
                    write_console(message)
                    log_handle.write(message)
                    log_handle.flush()
                    terminate_process(proc)
                    break

                if not drained:
                    time.sleep(float(args.poll_interval_sec))

        reader.join(timeout=5.0)

        checkpoint_after = load_checkpoint_counts(checkpoint_path)
        audio_after = count_audio_files(audio_dir)
        session_summary: Dict[str, object] = {
            "session_index": session_index,
            "started_at": datetime.fromtimestamp(session_start).isoformat(timespec="seconds"),
            "duration_sec": round(time.time() - session_start, 2),
            "stalled": stalled,
            "exit_code": proc.returncode,
            "limit": session_limit,
            "processed_before": session_checkpoint_before["processed"],
            "downloaded_before": session_checkpoint_before["downloaded"],
            "audio_before": session_audio_before,
            "processed_after": checkpoint_after["processed"],
            "downloaded_after": checkpoint_after["downloaded"],
            "audio_after": audio_after,
            "log_path": str(log_path),
        }

        session_summary["processed_gain"] = int(
            checkpoint_after["processed"] - session_checkpoint_before["processed"]
        )
        session_summary["downloaded_gain"] = int(
            checkpoint_after["downloaded"] - session_checkpoint_before["downloaded"]
        )
        session_summary["audio_gain"] = int(audio_after - session_audio_before)

        progress_made = any(
            int(session_summary[key]) > 0
            for key in ("processed_gain", "downloaded_gain", "audio_gain")
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
                "Downloader exceeded the allowed number of idle restarts. "
                f"See {run_summary_path} for details."
            )

        if progress_made:
            no_progress_sessions = 0
        else:
            no_progress_sessions += 1
            print(
                f"[resilient-download] session {session_index} finished with no new progress "
                f"(count={no_progress_sessions})."
            )
            if no_progress_sessions >= int(args.max_no_progress_sessions):
                print(
                    "[resilient-download] stopping after repeated no-progress sessions. "
                    "Remaining songs could not be downloaded in this run."
                )
                break

    final_checkpoint = load_checkpoint_counts(checkpoint_path)
    final_audio = count_audio_files(audio_dir)
    payload = {
        "catalog_rows": total_catalog_rows,
        "processed_rows": final_checkpoint["processed"],
        "downloaded_rows": final_checkpoint["downloaded"],
        "audio_files": final_audio,
        "remaining_rows": max(0, total_catalog_rows - final_checkpoint["processed"]),
        "forced_restart_count": forced_restart_count,
        "sessions": session_summaries,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    run_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nResilient download finished.")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
