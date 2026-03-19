# P2 Pipeline And Technical Debt Cleanup Report

Date: 2026-03-15

## Scope

This report covers the full `P2 - Pipeline and technical debt cleanup` section
from `docs/reports/implementation_todo_20260314.md`.

The targeted outcomes were:

- remove the broken `run_pipeline.py` dependency on the missing
  `src.audio_preprocessing.loudness_scanner`
- stop overclaiming true-peak behavior where the code only performs sample-peak
  protection
- remove or explicitly mark stale references to WKBSC, active mel-spectrogram
  assumptions, `.mp3`-only maintenance logic, and the old duration-only
  normalizer
- make helper and maintenance scripts consistent with the current processed
  `.wav` library

## Executive Summary

This section is complete.

The pipeline entrypoint has been simplified to the supported baseline path. The
broken inlined adaptive loudness-scan workflow is gone, `run_pipeline.py` now
delegates preprocessing to `scripts/run_audio_preprocessing.py`, and the
preprocessing package no longer advertises or exports the missing
`LoudnessScanner` path.

The preprocessing layer is now documented accurately:

- integrated loudness measurement is still based on ITU-R BS.1770 via
  `pyloudnorm`
- peak protection is now described consistently as a sample-peak ceiling
- the historical config/argument name `max_true_peak` is retained only for
  backward compatibility

The maintenance surface is also aligned to the current audio library:

- orphan cleanup is no longer `.mp3`-only
- metadata/audio checks use unified-schema `has_audio` semantics when available
- audio matching is done by basename so processed `.wav` files still match
  metadata that may have originated from legacy preview downloads
- the legacy duration-only `src/utils/audio_normalizer.py` path now delegates to
  the supported preprocessing pipeline instead of carrying a second, stale
  implementation

The documentation cleanup also landed:

- `README.md` no longer presents WKBSC as an active clustering option
- mel-spectrogram language was reduced to compatibility-only wording
- preprocessing docs no longer claim formal true-peak compliance
- a refreshed `docs/README.md` now reflects the supported baseline instead of
  the older mixed historical architecture

## What Changed

### 1. `run_pipeline.py` was simplified to the supported preprocessing path

Updated file:

- `run_pipeline.py`

The previous `preprocess` branch had a large inlined workflow that:

- imported the missing `src.audio_preprocessing.loudness_scanner`
- performed a dataset-wide scan pass
- selected an adaptive loudness target
- generated an old-style verification/report flow
- still assumed post-preprocessing `.mp3` files in parts of the logic

That code path no longer matched the supported workspace baseline and could not
run successfully because the scanner module was absent.

The new `run_pipeline.py` now:

- keeps a small `run()` helper for subprocess execution
- defines supported audio extensions once
- counts audio-backed metadata rows from the unified CSV correctly
- validates the audio library against `songs.csv` using the current schema
- delegates preprocessing to `scripts/run_audio_preprocessing.py`
- passes the supported baseline constants directly:
  - `29.0s`
  - `-14 LUFS`
  - `-1.0 dBFS` sample-peak ceiling
  - `22050 Hz`
- keeps the historical `process` skip token only as a compatibility no-op note

This is an intentional behavior change:

- the old adaptive loudness target-selection logic is no longer part of the
  main pipeline
- the supported baseline is now explicit and fixed instead of being implicitly
  re-derived from a scan step

### 2. The missing scanner reference was removed from the preprocessing package

Updated file:

- `src/audio_preprocessing/__init__.py`

Before this change, the package still had an optional export block for:

- `LoudnessScanner`
- `ScanResult`
- `DistributionAnalysis`

Even though the import was guarded, it preserved a stale contract around a
missing module.

That optional export path is now removed. The package surface is back to the
actual supported components:

- `AudioPreprocessor`
- `DurationHandler`
- `LoudnessNormalizer`

### 3. Preprocessing terminology was corrected from "true peak" to what the code actually does

Updated files:

- `src/audio_preprocessing/loudness_normalizer.py`
- `src/audio_preprocessing/processor.py`
- `scripts/run_audio_preprocessing.py`
- `docs/AUDIO_PREPROCESSING.md`
- `config/feature_vars.py`

The underlying DSP behavior did not change in this pass. The important change
is accuracy of description.

`loudness_normalizer.py` now:

- states clearly that it uses integrated loudness measurement plus a
  sample-peak safety ceiling
- explains that oversampled true-peak measurement is not implemented
- introduces `measure_sample_peak()` as the canonical method name
- keeps `measure_true_peak` as a backward-compatible alias

`processor.py` now:

- describes the pipeline step as a sample-peak safety cap
- stops calling the behavior EBU-R128-compliant true-peak limiting
- prints `Peak Ceiling: ... dBFS sample peak` in summaries

`scripts/run_audio_preprocessing.py` now:

- adds `--max-peak-db` as the primary CLI flag
- keeps `--max-true-peak` as an alias for compatibility

`config/feature_vars.py` now explicitly documents that:

- `baseline_max_true_peak_dbtp` keeps its historical name
- the actual implementation uses a sample-peak ceiling

`docs/AUDIO_PREPROCESSING.md` was rewritten to reflect the real behavior:

- supported runner and baseline invariants
- supported input formats
- WAV output policy
- sample-peak terminology instead of a true-peak claim
- legacy-name explanation for `max_true_peak`

### 4. The preprocessing implementation now matches the current audio-library assumptions better

Updated file:

- `src/audio_preprocessing/processor.py`

The processor now:

- scans `.wav`, `.mp3`, `.flac`, and `.m4a`
- sorts discovered files for stable traversal
- keeps the current `.wav` write-back behavior
- still removes too-short files and converts non-WAV inputs to WAV

This means the processor is no longer implicitly biased toward the older
`.mp3`-centric library layout.

### 5. The legacy normalizer path was reduced to a compatibility wrapper

Updated file:

- `src/utils/audio_normalizer.py`

Previously this file contained a completely separate duration-only MP3
normalizer implemented with `pydub`. That was a classic technical-debt trap:

- it encoded a different preprocessing contract
- it only understood MP3 input
- it did not reflect the supported loudness-normalized baseline

It now acts as a compatibility wrapper around the real supported preprocessing
pipeline. The wrapper:

- imports the baseline config
- instantiates `AudioPreprocessor`
- delegates directory processing to the supported path
- prints a deprecation-style notice telling callers to prefer
  `scripts/run_audio_preprocessing.py`

This keeps older imports from silently using stale behavior.

### 6. The orphan cleanup script was rebuilt around unified metadata and the `.wav` library

Updated file:

- `scripts/utilities/cleanup_orphaned_files.py`

This script had one of the most important stale assumptions in the repo:

- it only looked for `.mp3` files
- it compared raw filenames directly
- it did not understand the unified `has_audio` field

The new version now:

- supports `.wav`, `.mp3`, `.flac`, and `.m4a`
- resolves the active metadata CSV automatically
- uses `has_audio` when that column exists
- falls back to filename-based logic only for legacy CSVs
- matches metadata to audio by basename, not by exact extension
- adds `--dry-run`
- keeps `--auto-confirm` / `-y`
- reports extension breakdowns

This is materially better than the old count-only logic because basename-set
comparison can still detect drift even when raw counts happen to match.

That exact situation was observed in verification:

- audio-backed metadata rows: `5535`
- local audio files: `5535`
- yet one orphan basename still exists:
  - `Felipe Gordon - Happy Sunday .wav`

So the old `.mp3`-only count comparison would have missed a real inconsistency.

### 7. The problem-song inspection script now understands the unified schema

Updated file:

- `scripts/utilities/check_problem_songs.py`

This helper now:

- resolves the active CSV correctly
- checks unified `audio_basename` or `filename` rather than hardcoding `.mp3`
- searches the local library across `.wav`, `.mp3`, `.flac`, and `.m4a`
- prints the row's `has_audio` value when a metadata match exists

This makes the script useful again for the current workspace state.

### 8. The preprocessing report helper was updated for the current pipeline

Updated file:

- `scripts/utilities/create_preprocessing_report.py`

The old version was stale in multiple ways:

- `.mp3`-only file discovery
- direct filename matching that breaks after MP3-to-WAV conversion
- a pseudo "true peak" approximation in reporting

The new version now:

- scans all supported audio formats
- keys before/after measurements by basename
- records sample-peak values explicitly
- uses the supported preprocessing config defaults
- stores the preprocessor detail rows in the final report output

### 9. Plotting and high-level docs now explicitly treat mel-spectrogram support as legacy

Updated files:

- `scripts/visualization/ploting.py`
- `README.md`
- `docs/README.md`

`ploting.py` still knows how to visualize a mel-spectrogram array, but only as
legacy compatibility. The active extractor does not emit that feature by
default for the supported handcrafted baseline.

The script now:

- keeps the main active feature list as the default plotting set
- adds `melspectrogram` only when a matching `*_melspectrogram.npy` file exists

`README.md` was also cleaned up so it no longer:

- presents WKBSC as an active clustering method
- lists mel-spectrograms as part of the supported handcrafted baseline
- points visualization usage at the wrong path

The refreshed `docs/README.md` now:

- summarizes the current supported baseline
- points to the real preprocessing/feature/clustering entrypoints
- marks old historical paths as compatibility-only

## Files Changed

Implementation files:

- `run_pipeline.py`
- `scripts/run_audio_preprocessing.py`
- `src/audio_preprocessing/__init__.py`
- `src/audio_preprocessing/duration_handler.py`
- `src/audio_preprocessing/loudness_normalizer.py`
- `src/audio_preprocessing/processor.py`
- `src/utils/audio_normalizer.py`
- `scripts/utilities/cleanup_orphaned_files.py`
- `scripts/utilities/check_problem_songs.py`
- `scripts/utilities/create_preprocessing_report.py`
- `scripts/visualization/ploting.py`
- `config/feature_vars.py`

Documentation files:

- `README.md`
- `docs/AUDIO_PREPROCESSING.md`
- `docs/README.md`
- `docs/reports/implementation_todo_20260314.md`

## Verification

### 1. Python compilation

`py_compile` passed for the touched Python files:

- `run_pipeline.py`
- `scripts/run_audio_preprocessing.py`
- `src/audio_preprocessing/__init__.py`
- `src/audio_preprocessing/duration_handler.py`
- `src/audio_preprocessing/loudness_normalizer.py`
- `src/audio_preprocessing/processor.py`
- `src/utils/audio_normalizer.py`
- `scripts/utilities/cleanup_orphaned_files.py`
- `scripts/utilities/check_problem_songs.py`
- `scripts/utilities/create_preprocessing_report.py`
- `scripts/visualization/ploting.py`

An additional incremental `py_compile` re-check also passed after tightening the
`has_audio` logic in:

- `run_pipeline.py`
- `scripts/utilities/cleanup_orphaned_files.py`

### 2. Pipeline entrypoint import/runtime smoke check

Command run:

```bash
python run_pipeline.py --skip download preprocess extract plot cluster
```

Result:

- command exited cleanly
- this confirms the new top-level pipeline entrypoint imports and argument
  handling no longer fail because of the missing scanner path

### 3. Orphan cleanup dry-run

Command run:

```bash
python scripts/utilities/cleanup_orphaned_files.py --dry-run
```

Result:

- metadata CSV resolved correctly to `data/songs.csv`
- audio-backed metadata rows: `5535`
- detected local audio files: `5535` (`.wav:5535`)
- one orphan basename was still detected:
  - `Felipe Gordon - Happy Sunday .wav`
- no files were deleted in this pass

This verifies both the new all-format logic and the basename-based comparison.

### 4. Problem-song helper runtime check

Command run:

```bash
python scripts/utilities/check_problem_songs.py
```

Result:

- command completed successfully
- the script now reports unified metadata hits, `has_audio` state, audio-file
  presence, and feature-file presence under the current schema

### 5. Preprocessing runner live smoke test attempt

A live preprocessing smoke test was attempted on a copied sample WAV file using:

```bash
python scripts/run_audio_preprocessing.py --audio-dir output\\tmp_p2_preprocess_smoke --workers 1
```

Result:

- the command did not reach audio processing because the current sandbox
  environment does not have `librosa` installed
- failure mode:
  - `ModuleNotFoundError: No module named 'librosa'`

This is an environment dependency issue, not a syntax issue in the touched
files. Because of that missing package, I could not complete a live end-to-end
preprocessing execution in this sandbox.

## Behavior Changes Worth Calling Out

These changes are intentional and should be understood before future work:

1. `run_pipeline.py` no longer performs adaptive loudness target selection.
2. The supported baseline is now treated as fixed and explicit during
   preprocessing.
3. "True peak" should no longer be read literally in this codebase unless an
   oversampled implementation is added later.
4. Maintenance scripts now treat the processed `.wav` library as first-class
   instead of assuming an `.mp3` catalog.
5. The old duration-only normalizer path is no longer a separate behavior fork.

## Remaining Caveats

This P2 section is complete, but a few caveats remain:

- the current environment used for verification is missing `librosa`, so live
  preprocessing execution could not be fully exercised here
- one orphan local audio file was detected in dry-run mode but not deleted,
  because deleting local catalog files without explicit user confirmation would
  be too risky
- mel-spectrogram plotting support still exists as a legacy visualization path,
  intentionally, for older saved feature runs

## Completion Status

The `P2 - Pipeline and technical debt cleanup` checklist in
`docs/reports/implementation_todo_20260314.md` has been fully marked complete.
