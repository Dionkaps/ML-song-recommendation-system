"""
Dual audio preprocessor -- produces two preprocessed copies of each
downloaded audio file, one tuned for handcrafted feature extraction and
one tuned for pretrained self-supervised models.

Motivation
----------
Handcrafted features (MFCC, chroma, spectral statistics, Tzanetakis
rhythm + low-energy rate) have no internal loudness invariance and work
with a 22050 Hz sample rate that matches the Tzanetakis & Cook (2002)
convention and the derived MIR literature. They benefit from an EBU R128
-23 LUFS normalisation so that energy-sensitive statistics are comparable
across songs.

Pretrained self-supervised models (MERT at 24 kHz, EnCodecMAE at 24 kHz,
MusiCNN at 16 kHz) were trained on "in-the-wild" audio and apply their
own per-utterance normalization inside the feature extractor. Forcing a
LUFS target on their input is a distribution shift versus their
pretraining corpus. Resampling them from a 22050 Hz intermediate also
costs band-limited information -- 24 kHz preserves content up to 12 kHz
that MERT / EnCodecMAE actually use.

Storage trade-off
-----------------
Two preprocessed copies plus the original mp3 triples the disk footprint
of the audio stage. On the DGX (/storage/data4/*, no quota) this is
acceptable; on quota-constrained hosts the pretrained copy can be
reproduced on demand from the original.

Atomicity / cross-directory consistency
---------------------------------------
The two output directories (`audio_handcrafted/` and `audio_pretrained/`)
must stay in lock-step: either both contain the song, or neither does.
A source file is rejected (written nowhere) if any of:

  * decode failure  -- corrupted mp3 / unreadable container
  * too_short       -- strictly less than `min_duration` after decode
  * silent          -- integrated loudness is -inf on the 22050 Hz copy

Atomicity is enforced by writing to `<name>.wav.tmp` paths first and
renaming both only when both writes have completed successfully. A
crash between the two renames leaves two orphan `.tmp` files that are
cleaned up on the next run before any real writes happen.

Resume semantics
----------------
On re-invocation: if both final outputs exist the source is skipped.
If exactly one of the two outputs is present (prior crash mid-pair), the
orphan is deleted and the pair is re-produced from the original, so the
"both or neither" invariant is always re-established before feature
extraction runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from safety import assert_inside_workspace

try:
    from .duration_handler import DurationHandler
    from .loudness_normalizer import LoudnessNormalizer
except ImportError:  # pragma: no cover
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR.parent) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR.parent))
    from audio_preprocessing.duration_handler import DurationHandler
    from audio_preprocessing.loudness_normalizer import LoudnessNormalizer


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = WORKSPACE_DIR / "audio"
DEFAULT_HANDCRAFTED_DIR = WORKSPACE_DIR / "audio_handcrafted"
DEFAULT_PRETRAINED_DIR = WORKSPACE_DIR / "audio_pretrained"
DEFAULT_HANDCRAFTED_SAMPLE_RATE = 22050
DEFAULT_PRETRAINED_SAMPLE_RATE = 24000
DEFAULT_TARGET_DURATION = 29.0
DEFAULT_TARGET_LUFS = -23.0
DEFAULT_PEAK_DBFS = -1.0


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
logging.getLogger("audioread").setLevel(logging.ERROR)
logging.getLogger("librosa").setLevel(logging.ERROR)


class _SuppressStderr:
    """Silence C-level warnings from mpg123 / libsndfile during decode."""

    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")


@dataclass
class _PerFileResult:
    source: str
    status: str  # "success" | "skipped_existing" | "removed" | "error"
    reason: str | None = None
    original_duration: float = 0.0
    final_duration: float = 0.0
    original_lufs: float | None = None
    final_lufs: float | None = None
    original_peak_db: float | None = None
    handcrafted_peak_db: float | None = None
    pretrained_peak_db: float | None = None
    gain_applied_db: float = 0.0
    # `default_factory=list` so every instance gets its own list and
    # `result.actions.append(...)` never hits an AttributeError on None.
    actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        # JSON cannot serialise inf / NaN. Silent tracks and pure-zero
        # signals produce -inf on the LUFS / peak fields, so coerce any
        # non-finite value to None here to keep the summary JSON valid.
        def _finite(value: float | None) -> float | None:
            if value is None:
                return None
            if not np.isfinite(value):
                return None
            return round(float(value), 3)

        return {
            "source": self.source,
            "status": self.status,
            "reason": self.reason,
            "original_duration": round(self.original_duration, 3),
            "final_duration": round(self.final_duration, 3),
            "original_lufs": _finite(self.original_lufs),
            "final_lufs": _finite(self.final_lufs),
            "original_peak_db": _finite(self.original_peak_db),
            "handcrafted_peak_db": _finite(self.handcrafted_peak_db),
            "pretrained_peak_db": _finite(self.pretrained_peak_db),
            "gain_applied_db": round(self.gain_applied_db, 3),
            "actions": list(self.actions),
        }


class DualAudioPreprocessor:
    """Produce two preprocessed wav copies per input, with locked rejection.

    The source directory is read-only; the two output directories are
    kept in sync (see module docstring for the atomicity contract).
    """

    def __init__(
        self,
        target_duration: float = DEFAULT_TARGET_DURATION,
        handcrafted_sample_rate: int = DEFAULT_HANDCRAFTED_SAMPLE_RATE,
        pretrained_sample_rate: int = DEFAULT_PRETRAINED_SAMPLE_RATE,
        target_lufs: float = DEFAULT_TARGET_LUFS,
        max_peak_dbfs: float = DEFAULT_PEAK_DBFS,
    ):
        if target_duration <= 0:
            raise ValueError("target_duration must be positive")
        if handcrafted_sample_rate <= 0 or pretrained_sample_rate <= 0:
            raise ValueError("sample rates must be positive")
        if max_peak_dbfs > 0:
            raise ValueError("max_peak_dbfs must be <= 0 dBFS")

        self.target_duration = target_duration
        self.handcrafted_sample_rate = handcrafted_sample_rate
        self.pretrained_sample_rate = pretrained_sample_rate
        self.target_lufs = target_lufs
        self.max_peak_dbfs = max_peak_dbfs
        self._max_peak_linear = 10.0 ** (max_peak_dbfs / 20.0)

        self.duration_handler = DurationHandler(
            target_duration=target_duration,
            crop_location="center",
        )
        self.loudness_normalizer = LoudnessNormalizer(
            target_lufs=target_lufs,
            max_true_peak=max_peak_dbfs,
        )

    # ------------------------------------------------------------------
    # Single-file processing
    # ------------------------------------------------------------------

    def _target_stem(self, source_path: Path) -> str:
        return source_path.stem

    def _output_paths(
        self,
        source_path: Path,
        handcrafted_dir: Path,
        pretrained_dir: Path,
    ) -> tuple[Path, Path]:
        stem = self._target_stem(source_path)
        return handcrafted_dir / f"{stem}.wav", pretrained_dir / f"{stem}.wav"

    def _reconcile_orphan_outputs(
        self,
        handcrafted_out: Path,
        pretrained_out: Path,
    ) -> bool:
        """If exactly one output exists, delete it so the pair re-syncs.

        Returns True when both exist *after* reconciliation (i.e. nothing
        to do -- caller may skip). Returns False when the caller must
        produce fresh outputs.
        """
        h_exists = handcrafted_out.exists()
        p_exists = pretrained_out.exists()

        if h_exists and p_exists:
            return True
        if h_exists and not p_exists:
            logger.warning(
                "Orphan handcrafted output without pretrained pair: %s -- removing",
                handcrafted_out.name,
            )
            handcrafted_out.unlink()
        elif p_exists and not h_exists:
            logger.warning(
                "Orphan pretrained output without handcrafted pair: %s -- removing",
                pretrained_out.name,
            )
            pretrained_out.unlink()
        # Clear any leftover .tmp files from a prior crash
        for path in (handcrafted_out, pretrained_out):
            tmp = path.with_suffix(path.suffix + ".tmp")
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        return False

    def _write_atomic_pair(
        self,
        handcrafted_out: Path,
        pretrained_out: Path,
        y_hand: np.ndarray,
        y_pre: np.ndarray,
    ) -> None:
        """Write both wavs via .tmp files, then rename together.

        If either write fails, both .tmp files are removed so the next
        run can start from a clean slate. Renames are sequential (POSIX
        has no atomic multi-file rename), so a crash between the two
        renames leaves exactly one final file; `_reconcile_orphan_outputs`
        cleans that up on the next invocation.
        """
        handcrafted_tmp = handcrafted_out.with_suffix(handcrafted_out.suffix + ".tmp")
        pretrained_tmp = pretrained_out.with_suffix(pretrained_out.suffix + ".tmp")

        try:
            handcrafted_out.parent.mkdir(parents=True, exist_ok=True)
            pretrained_out.parent.mkdir(parents=True, exist_ok=True)

            # Explicit format="WAV" because the .tmp suffix hides the real
            # extension and soundfile refuses to guess.
            sf.write(
                str(handcrafted_tmp), y_hand, self.handcrafted_sample_rate,
                subtype="PCM_16", format="WAV",
            )
            sf.write(
                str(pretrained_tmp), y_pre, self.pretrained_sample_rate,
                subtype="PCM_16", format="WAV",
            )

            os.replace(handcrafted_tmp, handcrafted_out)
            os.replace(pretrained_tmp, pretrained_out)
        except Exception:
            for tmp in (handcrafted_tmp, pretrained_tmp):
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
            # Also roll back the first rename if it succeeded before the second failed
            if handcrafted_out.exists() and not pretrained_out.exists() and not pretrained_tmp.exists():
                try:
                    handcrafted_out.unlink()
                except OSError:
                    pass
            raise

    @staticmethod
    def _sample_peak_db(y: np.ndarray) -> float:
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak <= 0:
            return -np.inf
        return 20.0 * np.log10(peak)

    def _apply_peak_cap(self, y: np.ndarray) -> tuple[np.ndarray, bool]:
        """Scale y so that max|y| <= max_peak_linear. Returns (y, was_capped)."""
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        # The `_max_peak_linear > 0` guard is defensive: a pathological
        # `max_peak_dbfs = -inf` would yield 0, and capping to 0 would
        # silence the entire signal, which is never what the caller wants.
        if peak > self._max_peak_linear and self._max_peak_linear > 0:
            return y * (self._max_peak_linear / peak), True
        return y, False

    def process_file(
        self,
        source_path: str,
        handcrafted_dir: str | Path = DEFAULT_HANDCRAFTED_DIR,
        pretrained_dir: str | Path = DEFAULT_PRETRAINED_DIR,
    ) -> Dict:
        source = Path(source_path)
        handcrafted_dir = assert_inside_workspace(handcrafted_dir, "handcrafted_audio_dir")
        pretrained_dir = assert_inside_workspace(pretrained_dir, "pretrained_audio_dir")

        result = _PerFileResult(source=source.name, status="success")

        handcrafted_out, pretrained_out = self._output_paths(
            source, handcrafted_dir, pretrained_dir,
        )

        if self._reconcile_orphan_outputs(handcrafted_out, pretrained_out):
            result.status = "skipped_existing"
            result.actions.append("skipped_existing")
            return result.to_dict()

        # ---- Load original at its native sample rate --------------------
        try:
            with _SuppressStderr():
                y_orig, sr_orig = librosa.load(str(source), sr=None, mono=True)
        except Exception as exc:
            result.status = "error"
            result.reason = f"decode_error: {exc}"
            logger.error("Decode failure for %s: %s", source.name, exc)
            return result.to_dict()

        if y_orig.size == 0 or sr_orig <= 0:
            result.status = "error"
            result.reason = "decode_error: empty_signal"
            return result.to_dict()

        result.original_duration = len(y_orig) / sr_orig

        # ---- Duration gating + center crop on the original signal -------
        y_window, duration_action = self.duration_handler.process(y_orig, sr_orig)
        if duration_action == "too_short":
            result.status = "removed"
            result.reason = "too_short"
            result.actions.append("rejected_too_short")
            return result.to_dict()

        if duration_action == "cropped":
            result.actions.append("center_cropped")
        result.final_duration = len(y_window) / sr_orig

        # ---- Resample into both target rates (polyphase via librosa) ----
        try:
            if sr_orig == self.handcrafted_sample_rate:
                y_hand = y_window.astype(np.float32)
            else:
                y_hand = librosa.resample(
                    y_window, orig_sr=sr_orig,
                    target_sr=self.handcrafted_sample_rate,
                ).astype(np.float32)

            if sr_orig == self.pretrained_sample_rate:
                y_pre = y_window.astype(np.float32)
            else:
                y_pre = librosa.resample(
                    y_window, orig_sr=sr_orig,
                    target_sr=self.pretrained_sample_rate,
                ).astype(np.float32)
        except Exception as exc:
            result.status = "error"
            result.reason = f"resample_error: {exc}"
            logger.error("Resample failure for %s: %s", source.name, exc)
            return result.to_dict()

        result.original_peak_db = self._sample_peak_db(y_window)

        # ---- Handcrafted branch: LUFS normalisation + peak cap ----------
        y_hand_out, norm_stats = self.loudness_normalizer.process(
            y_hand, self.handcrafted_sample_rate,
        )
        # Mirror normalizer stats onto the result
        result.original_lufs = norm_stats.get("original_lufs")
        # Silent or pathological track (integrated loudness is -inf or NaN):
        # reject symmetrically from both output directories. Using
        # `not np.isfinite` rather than `np.isinf` catches the NaN case too,
        # which pyloudnorm can return on degenerate inputs.
        if result.original_lufs is None or not np.isfinite(result.original_lufs):
            result.status = "removed"
            result.reason = "silent"
            result.actions.append("rejected_silent")
            return result.to_dict()

        result.final_lufs = norm_stats.get("final_lufs")
        result.gain_applied_db = float(norm_stats.get("gain_applied_db", 0.0) or 0.0)
        result.actions.extend(norm_stats.get("actions", []))
        result.handcrafted_peak_db = self._sample_peak_db(y_hand_out)

        # ---- Pretrained branch: peak-only safety cap (no LUFS) ----------
        y_pre_out, pre_capped = self._apply_peak_cap(y_pre)
        if pre_capped:
            result.actions.append("pretrained_peak_limited")
        result.pretrained_peak_db = self._sample_peak_db(y_pre_out)

        # ---- Atomic paired write ----------------------------------------
        try:
            self._write_atomic_pair(
                handcrafted_out, pretrained_out, y_hand_out, y_pre_out,
            )
        except Exception as exc:
            result.status = "error"
            result.reason = f"write_error: {exc}"
            logger.error("Write failure for %s: %s", source.name, exc)
            return result.to_dict()

        result.actions.append("wrote_pair")
        return result.to_dict()

    # ------------------------------------------------------------------
    # Directory processing
    # ------------------------------------------------------------------

    def process_directory(
        self,
        source_dir: str | Path = DEFAULT_SOURCE_DIR,
        handcrafted_dir: str | Path = DEFAULT_HANDCRAFTED_DIR,
        pretrained_dir: str | Path = DEFAULT_PRETRAINED_DIR,
        max_workers: int | None = None,
        summary_path: str | Path | None = None,
        return_details: bool = False,
    ) -> Dict:
        source_dir = Path(source_dir)
        handcrafted_dir = assert_inside_workspace(handcrafted_dir, "handcrafted_audio_dir")
        pretrained_dir = assert_inside_workspace(pretrained_dir, "pretrained_audio_dir")
        if summary_path is not None:
            summary_path = assert_inside_workspace(summary_path, "preprocess_summary")

        if not source_dir.exists():
            logger.error("Source directory not found: %s", source_dir)
            return {"error": "source_dir_missing", "source_dir": str(source_dir)}

        # Collect candidates per stem so we can deduplicate when both
        # `foo.mp3` and `foo.wav` coexist (a known scenario when a previous
        # destructive run left .wav files alongside the original mp3
        # downloads). Without dedup, two workers would target identical
        # output paths and race for the same `.wav.tmp` file.
        ext_priority = {ext: rank for rank, ext in enumerate((
            ".mp3", ".m4a", ".flac", ".wav",  # mp3 first: pristine source
        ))}
        by_stem: dict[str, Path] = {}
        ignored_duplicates: list[str] = []
        legacy_wav_count = 0

        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            for path in source_dir.glob(f"*{ext}"):
                suffix_lc = path.suffix.lower()
                if suffix_lc == ".wav":
                    legacy_wav_count += 1
                stem = path.stem
                if stem not in by_stem:
                    by_stem[stem] = path
                    continue
                existing = by_stem[stem]
                cur_rank = ext_priority.get(suffix_lc, 99)
                old_rank = ext_priority.get(existing.suffix.lower(), 99)
                if cur_rank < old_rank:
                    ignored_duplicates.append(existing.name)
                    by_stem[stem] = path
                else:
                    ignored_duplicates.append(path.name)

        files = sorted(by_stem.values())

        if not files:
            logger.warning("No audio files found in %s", source_dir)
            return {"total": 0, "source_dir": str(source_dir)}

        if ignored_duplicates:
            logger.warning(
                "Source directory contains %d duplicate stem(s); preferred the "
                "higher-priority extension. Ignored: %s%s",
                len(ignored_duplicates),
                ", ".join(ignored_duplicates[:5]),
                f", ... ({len(ignored_duplicates) - 5} more)" if len(ignored_duplicates) > 5 else "",
            )

        if legacy_wav_count > 0:
            logger.warning(
                "%d .wav file(s) detected in %s. If these are leftovers from a "
                "previous destructive preprocessing run (already 22 kHz / -14 LUFS), "
                "the pretrained branch will upsample them 22050->%d Hz, which is "
                "lossless but adds a redundant resample step. For ideal pretrained "
                "embeddings, delete the workspace's audio/ and re-download from "
                "Deezer originals (mp3, native 44.1 kHz).",
                legacy_wav_count, source_dir, self.pretrained_sample_rate,
            )

        if max_workers is None:
            default_cap = int(os.environ.get("MAX_WORKERS", 16))
            max_workers = min(cpu_count(), default_cap)

        handcrafted_dir.mkdir(parents=True, exist_ok=True)
        pretrained_dir.mkdir(parents=True, exist_ok=True)

        stats: Dict[str, object] = {
            "total": len(files),
            "source_dir": str(source_dir.resolve()),
            "handcrafted_dir": str(handcrafted_dir.resolve()),
            "pretrained_dir": str(pretrained_dir.resolve()),
            "handcrafted_sample_rate": self.handcrafted_sample_rate,
            "pretrained_sample_rate": self.pretrained_sample_rate,
            "target_lufs": self.target_lufs,
            "target_duration": self.target_duration,
            "max_peak_dbfs": self.max_peak_dbfs,
            "processed": 0,
            "skipped_existing": 0,
            "removed_too_short": 0,
            "removed_silent": 0,
            "errors": 0,
            "peak_limited_handcrafted": 0,
            "peak_limited_pretrained": 0,
            "rejected_files": [],
            "error_files": [],
            "details": [] if return_details else None,
        }

        print(f"\n{'=' * 60}")
        print("DUAL AUDIO PREPROCESSING")
        print(f"{'=' * 60}")
        print(f"Source:               {source_dir}")
        print(f"Handcrafted output:   {handcrafted_dir}  "
              f"({self.handcrafted_sample_rate} Hz, {self.target_lufs} LUFS)")
        print(f"Pretrained output:    {pretrained_dir}  "
              f"({self.pretrained_sample_rate} Hz, no LUFS norm)")
        print(f"Target duration:      {self.target_duration}s (center crop)")
        print(f"Peak ceiling:         {self.max_peak_dbfs} dBFS (sample peak)")
        print(f"Files to process:     {len(files)}")
        print(f"Workers:              {max_workers}")
        print(f"{'=' * 60}\n")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self.process_file,
                    str(path),
                    str(handcrafted_dir),
                    str(pretrained_dir),
                ): path
                for path in files
            }

            for future in tqdm(
                as_completed(future_to_file),
                total=len(files),
                desc="Preprocessing",
            ):
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - worker crash
                    path = future_to_file[future]
                    logger.error("Worker crashed on %s: %s", path.name, exc)
                    result = {
                        "source": path.name,
                        "status": "error",
                        "reason": f"worker_crash: {exc}",
                        "actions": [],
                    }

                if return_details:
                    stats["details"].append(result)

                status = result.get("status", "error")
                actions = result.get("actions") or []

                if status == "success":
                    stats["processed"] += 1
                    if "peak_limited" in actions:
                        stats["peak_limited_handcrafted"] += 1
                    if "pretrained_peak_limited" in actions:
                        stats["peak_limited_pretrained"] += 1
                elif status == "skipped_existing":
                    stats["skipped_existing"] += 1
                elif status == "removed":
                    reason = result.get("reason", "removed")
                    stats["rejected_files"].append({
                        "source": result.get("source"),
                        "reason": reason,
                    })
                    if reason == "too_short":
                        stats["removed_too_short"] += 1
                    elif reason == "silent":
                        stats["removed_silent"] += 1
                else:  # error
                    stats["errors"] += 1
                    stats["error_files"].append({
                        "source": result.get("source"),
                        "reason": result.get("reason"),
                    })

        print(f"\n{'=' * 60}")
        print("PREPROCESSING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total files:                 {stats['total']}")
        print(f"Processed (wrote pair):      {stats['processed']}")
        print(f"Skipped (already present):   {stats['skipped_existing']}")
        print(f"Rejected (too short):        {stats['removed_too_short']}")
        print(f"Rejected (silent):           {stats['removed_silent']}")
        print(f"Errors:                      {stats['errors']}")
        print(f"Peak-limited handcrafted:    {stats['peak_limited_handcrafted']}")
        print(f"Peak-limited pretrained:     {stats['peak_limited_pretrained']}")
        print(f"{'=' * 60}\n")

        if summary_path is not None:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_payload = {k: v for k, v in stats.items() if k != "details"}
            summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Produce two preprocessed audio copies per source file: one for "
            "handcrafted feature extraction (22 kHz, -23 LUFS, center crop) "
            "and one for pretrained self-supervised models (24 kHz, no LUFS). "
            "Rejections are synchronised across the two output directories."
        ),
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR),
                        help="Directory containing the downloaded source audio (default: audio/)")
    parser.add_argument("--handcrafted-dir", default=str(DEFAULT_HANDCRAFTED_DIR),
                        help="Output directory for handcrafted-feature preprocessing (default: audio_handcrafted/)")
    parser.add_argument("--pretrained-dir", default=str(DEFAULT_PRETRAINED_DIR),
                        help="Output directory for pretrained-model preprocessing (default: audio_pretrained/)")
    parser.add_argument("--lufs", type=float, default=DEFAULT_TARGET_LUFS,
                        help=f"Target LUFS for the handcrafted copy (default: {DEFAULT_TARGET_LUFS}, EBU R128)")
    parser.add_argument("--duration", type=float, default=DEFAULT_TARGET_DURATION,
                        help="Target duration in seconds (center-cropped)")
    parser.add_argument("--peak", type=float, default=DEFAULT_PEAK_DBFS,
                        help="Sample-peak ceiling in dBFS (applied to both copies)")
    parser.add_argument("--handcrafted-sample-rate", type=int, default=DEFAULT_HANDCRAFTED_SAMPLE_RATE,
                        help="Sample rate for the handcrafted copy")
    parser.add_argument("--pretrained-sample-rate", type=int, default=DEFAULT_PRETRAINED_SAMPLE_RATE,
                        help="Sample rate for the pretrained copy")
    parser.add_argument("--workers", type=int, help="Parallel worker count")
    parser.add_argument("--summary-path", default="",
                        help="Optional JSON summary output path")
    parser.add_argument("--details", action="store_true",
                        help="Keep per-file details in memory (large runs)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = DualAudioPreprocessor(
        target_duration=args.duration,
        handcrafted_sample_rate=args.handcrafted_sample_rate,
        pretrained_sample_rate=args.pretrained_sample_rate,
        target_lufs=args.lufs,
        max_peak_dbfs=args.peak,
    )
    processor.process_directory(
        source_dir=args.source_dir,
        handcrafted_dir=args.handcrafted_dir,
        pretrained_dir=args.pretrained_dir,
        max_workers=args.workers,
        summary_path=args.summary_path or None,
        return_details=args.details,
    )


if __name__ == "__main__":
    main()
