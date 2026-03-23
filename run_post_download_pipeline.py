from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.experiment_profiles import build_canonical_baseline_contract  # noqa: E402
from scripts.utilities.run_full_benchmark_rerun import (  # noqa: E402
    build_cleanup_targets,
    build_steps,
    ensure_dir,
    remove_path,
    verify_download_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full post-download pipeline from audio preprocessing through "
            "the thesis benchmark, with timestamped per-step logs and a generated "
            "Markdown report."
        )
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run every pipeline step.",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional suffix added to the timestamped log folder name.",
    )
    parser.add_argument(
        "--include-plot",
        action="store_true",
        help="Include the optional bulk plotting stage.",
    )
    parser.add_argument(
        "--log-root",
        default="docs/reports/run_logs",
        help="Root directory under which the timestamped run folder will be created.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final pipeline summary as JSON instead of a human-readable message.",
    )
    return parser.parse_args()


class OrchestratorLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("a", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message)
        stamp = datetime.now().isoformat(timespec="seconds")
        if message:
            self._handle.write(f"[{stamp}] {message}\n")
        else:
            self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def normalize_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_capture(command: List[str]) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception as exc:
        return {
            "command": command,
            "error": str(exc),
        }

    return {
        "command": command,
        "returncode": int(result.returncode),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def collect_git_context() -> Dict[str, Any]:
    return {
        "branch": run_capture(["git", "branch", "--show-current"]),
        "head": run_capture(["git", "rev-parse", "--short", "HEAD"]),
        "status_short": run_capture(["git", "status", "--short"]),
    }


def command_display(command: List[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def build_post_download_steps(
    python_executable: str,
    include_plot: bool,
    benchmark_output_dir: Path,
    experiment_root: Path,
    run_root: Path,
    report_command: List[str],
) -> List[Dict[str, Any]]:
    args = Namespace(
        python=python_executable,
        preserve_download_state=True,
        include_plot=include_plot,
    )
    steps = build_steps(
        args=args,
        benchmark_output_dir=benchmark_output_dir,
        experiment_root=experiment_root,
    )

    for step in steps:
        if str(step.get("name", "")).endswith("run_audio_preprocessing"):
            command = [str(part) for part in step["command"]]
            if "--details" not in command:
                command.append("--details")
            if "--details-output" not in command:
                command.extend(
                    [
                        "--details-output",
                        str(run_root / "01_run_audio_preprocessing_details.json"),
                    ]
                )
            step["command"] = command

    steps.append(
        {
            "name": f"{len(steps) + 1:02d}_generate_detailed_report",
            "command": [str(part) for part in report_command],
        }
    )
    return steps


def run_logged_step(
    name: str,
    command: List[str],
    log_dir: Path,
    orchestrator: OrchestratorLogger,
) -> Dict[str, Any]:
    log_path = log_dir / f"{name}.log"
    started_at = datetime.now().isoformat(timespec="seconds")
    started_perf = perf_counter()
    display = command_display(command)

    orchestrator.log("")
    orchestrator.log(f"[{name}] Starting")
    orchestrator.log(f"$ {display}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {display}\n\n")
        handle.flush()

        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        process.stdout.close()
        returncode = process.wait()

    ended_at = datetime.now().isoformat(timespec="seconds")
    duration_sec = round(perf_counter() - started_perf, 2)
    orchestrator.log(f"[{name}] Exit code: {returncode} | duration: {duration_sec:.2f}s")
    orchestrator.log(f"[{name}] Log: {log_path}")

    if returncode != 0:
        tail_lines = log_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()[-40:]
        if tail_lines:
            orchestrator.log(f"[{name}] Last log lines:")
            for line in tail_lines:
                orchestrator.log(line)
        raise subprocess.CalledProcessError(returncode, command)

    return {
        "name": name,
        "command": [str(part) for part in command],
        "log_path": str(log_path),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_sec": duration_sec,
        "returncode": int(returncode),
    }


def build_scope_summary(include_plot: bool) -> str:
    plot_note = (
        "The optional plotting stage was included in this run."
        if include_plot
        else "The optional plotting stage was skipped to keep the rerun focused on the benchmark path."
    )
    return (
        "This report summarizes a post-download rerun that preserved the existing "
        "downloaded audio and metadata inputs, cleaned only derived artifacts, "
        "reran audio preprocessing and handcrafted feature extraction, executed "
        "the clustering/profile evaluation stages, and finished with the thesis "
        "clustering benchmark. "
        + plot_note
    )


def build_summary(
    *,
    args: argparse.Namespace,
    run_root: Path,
    run_context_path: Path,
    baseline_contract_path: Path,
    planned_steps_path: Path,
    cleanup_summary_path: Path,
    download_verification_path: Path,
    download_verification: Dict[str, Any],
    cleanup_results: List[Dict[str, Any]],
    step_results: List[Dict[str, Any]],
    benchmark_output_dir: Path,
    experiment_root: Path,
    orchestrator_log_path: Path,
    preprocessing_details_path: Path,
    report_path: Path,
    status: str,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    notes = [
        "This run preserved the download-stage inputs and started from audio preprocessing.",
        "Each pipeline stage wrote its own timestamped step log under this run folder.",
        "The preprocessing step was executed with --details and writes a UTF-8 per-file JSON artifact inside the run folder.",
    ]
    if args.include_plot:
        notes.append("The optional plotting stage was included.")
    else:
        notes.append("The optional plotting stage was skipped.")

    return {
        "status": status,
        "project_root": str(PROJECT_ROOT),
        "python": str(args.python),
        "run_root": str(run_root),
        "rerun_mode": "post_download_pipeline",
        "scope_summary": build_scope_summary(include_plot=bool(args.include_plot)),
        "download_inputs_preserved": True,
        "included_plot_stage": bool(args.include_plot),
        "run_label": str(args.run_label),
        "run_context_path": str(run_context_path),
        "canonical_baseline_contract_path": str(baseline_contract_path),
        "planned_steps_path": str(planned_steps_path),
        "cleanup_summary_path": str(cleanup_summary_path),
        "download_verification_path": str(download_verification_path),
        "download_verification": download_verification,
        "cleanup_targets": cleanup_results,
        "steps": step_results,
        "benchmark_output_dir": str(benchmark_output_dir),
        "experiment_run_root": str(experiment_root),
        "orchestrator_log_path": str(orchestrator_log_path),
        "preprocessing_details_path": str(preprocessing_details_path),
        "report_path": str(report_path),
        "benchmark_rerun_notes": notes,
        "error": error,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_suffix = f"_{args.run_label.strip()}" if args.run_label.strip() else ""
    log_root = normalize_path(args.log_root)
    run_root = ensure_dir(log_root / f"post_download_pipeline_{stamp}{label_suffix}")
    orchestrator_log_path = run_root / "orchestrator.log"
    orchestrator = OrchestratorLogger(orchestrator_log_path)

    report_path = run_root / "detailed_report.md"
    summary_path = run_root / "rerun_summary.json"
    cleanup_summary_path = run_root / "cleanup_summary.json"
    run_context_path = run_root / "run_context.json"
    baseline_contract_path = run_root / "canonical_baseline_contract.json"
    planned_steps_path = run_root / "planned_steps.json"
    download_verification_path = run_root / "download_verification.json"
    preprocessing_details_path = run_root / "01_run_audio_preprocessing_details.json"

    output_root = PROJECT_ROOT / "output"
    metrics_root = output_root / "metrics"
    experiment_root = output_root / "experiment_runs_taxonomy"
    benchmark_output_dir = metrics_root / f"thesis_benchmark_full_rerun_{stamp}"

    report_command = [
        args.python,
        "scripts/utilities/generate_full_rerun_report.py",
        "--run-root",
        str(run_root),
        "--output-path",
        str(report_path),
    ]
    planned_steps = build_post_download_steps(
        python_executable=str(args.python),
        include_plot=bool(args.include_plot),
        benchmark_output_dir=benchmark_output_dir,
        experiment_root=experiment_root,
        run_root=run_root,
        report_command=report_command,
    )

    download_verification: Dict[str, Any] = {}
    cleanup_results: List[Dict[str, Any]] = []
    step_results: List[Dict[str, Any]] = []
    status = "failed"
    error: Optional[Dict[str, Any]] = None

    try:
        baseline_contract = build_canonical_baseline_contract()
        write_json(baseline_contract_path, baseline_contract)

        run_context = {
            "run_id": run_root.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(PROJECT_ROOT),
            "cwd": str(PROJECT_ROOT),
            "python_executable": str(args.python),
            "command_line": sys.argv,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
            "git": collect_git_context(),
            "planned_steps": planned_steps,
        }
        write_json(run_context_path, run_context)
        write_json(planned_steps_path, {"steps": planned_steps})

        orchestrator.log(f"Run root: {run_root}")
        orchestrator.log(f"Run context: {run_context_path}")
        orchestrator.log(f"Baseline contract: {baseline_contract_path}")
        orchestrator.log(f"Planned steps: {planned_steps_path}")

        download_verification = verify_download_state()
        write_json(download_verification_path, download_verification)
        orchestrator.log(f"Download verification: {download_verification_path}")

        orchestrator.log("")
        orchestrator.log("[cleanup] Removing derived artifacts while preserving download inputs...")
        cleanup_targets = build_cleanup_targets(preserve_download_state=True)
        cleanup_results = [remove_path(path) for path in cleanup_targets]
        write_json(cleanup_summary_path, {"targets": cleanup_results})
        for item in cleanup_results:
            state = "removed" if item["existed"] else "missing"
            orchestrator.log(f"[cleanup] {state}: {item['path']}")

        ensure_dir(output_root)
        ensure_dir(metrics_root)
        ensure_dir(experiment_root)

        for step in planned_steps[:-1]:
            step_results.append(
                run_logged_step(
                    name=str(step["name"]),
                    command=[str(part) for part in step["command"]],
                    log_dir=run_root,
                    orchestrator=orchestrator,
                )
            )

        draft_summary = build_summary(
            args=args,
            run_root=run_root,
            run_context_path=run_context_path,
            baseline_contract_path=baseline_contract_path,
            planned_steps_path=planned_steps_path,
            cleanup_summary_path=cleanup_summary_path,
            download_verification_path=download_verification_path,
            download_verification=download_verification,
            cleanup_results=cleanup_results,
            step_results=step_results,
            benchmark_output_dir=benchmark_output_dir,
            experiment_root=experiment_root,
            orchestrator_log_path=orchestrator_log_path,
            preprocessing_details_path=preprocessing_details_path,
            report_path=report_path,
            status="running_report_generation",
            error=None,
        )
        write_json(summary_path, draft_summary)

        report_step = planned_steps[-1]
        step_results.append(
            run_logged_step(
                name=str(report_step["name"]),
                command=[str(part) for part in report_step["command"]],
                log_dir=run_root,
                orchestrator=orchestrator,
            )
        )

        status = "completed"

    except subprocess.CalledProcessError as exc:
        failed_step = planned_steps[len(step_results)] if len(step_results) < len(planned_steps) else None
        error = {
            "type": "CalledProcessError",
            "returncode": int(exc.returncode),
            "command": [str(part) for part in exc.cmd],
            "failed_step": failed_step["name"] if failed_step else None,
        }
        orchestrator.log("")
        orchestrator.log("Pipeline execution stopped because one step failed.")
    except Exception as exc:
        failed_step = planned_steps[len(step_results)] if len(step_results) < len(planned_steps) else None
        error = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "failed_step": failed_step["name"] if failed_step else None,
        }
        orchestrator.log("")
        orchestrator.log(f"Unexpected error: {exc}")
    finally:
        final_summary = build_summary(
            args=args,
            run_root=run_root,
            run_context_path=run_context_path,
            baseline_contract_path=baseline_contract_path,
            planned_steps_path=planned_steps_path,
            cleanup_summary_path=cleanup_summary_path,
            download_verification_path=download_verification_path,
            download_verification=download_verification,
            cleanup_results=cleanup_results,
            step_results=step_results,
            benchmark_output_dir=benchmark_output_dir,
            experiment_root=experiment_root,
            orchestrator_log_path=orchestrator_log_path,
            preprocessing_details_path=preprocessing_details_path,
            report_path=report_path,
            status=status,
            error=error,
        )
        write_json(summary_path, final_summary)
        orchestrator.log("")
        orchestrator.log(f"Run summary: {summary_path}")
        orchestrator.log(f"Detailed report: {report_path}")
        orchestrator.close()

    payload = {
        "status": status,
        "run_root": str(run_root),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Post-download pipeline {status}.")
        print(f"Run root: {run_root}")
        print(f"Run summary: {summary_path}")
        print(f"Detailed report: {report_path}")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
