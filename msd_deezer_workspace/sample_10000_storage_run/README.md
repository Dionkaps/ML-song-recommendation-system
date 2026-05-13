# sample_10000_storage_run

Isolated benchmark workspace for a 10,000-song storage sample.

This folder is intended to be run on the DGX, where the production downloaded
audio directory should exist as:

```text
msd_deezer_workspace/audio/
```

The local laptop checkout has no downloaded audio, so audio copying must happen
on the DGX.

These helper files must be committed and pushed, or copied manually to the DGX,
before the DGX commands below can run.

Intended layout:

- `audio_10000/`: exactly 10,000 copied source audio files for the benchmark.
- `audio_handcrafted/`: handcrafted preprocessing outputs.
- `audio_pretrained/`: pretrained-model preprocessing outputs.
- `features/`: handcrafted feature outputs.
- `pretrained_embeddings/`: pretrained embedding outputs.
- `data/`: manifests and benchmark metadata.
- `cache/`: benchmark-local caches only.
- `logs/`: benchmark-local logs only.

## DGX Usage

Requires DGX/Linux bash with GNU coreutils.

```bash
source /opt/anaconda3/bin/activate
conda activate /storage/data4/up1072603/conda_envs/msdrec
cd /storage/data4/up1072603/projects/ML-song-recommendation-system/msd_deezer_workspace

# Optional cleanup on DGX, only if these generated folders exist there.
# Do not use rm -rf sample_*.
for name in sample_10_storage_run sample_100_storage_run; do
  target="$(realpath -m "$PWD/$name")"
  case "$target" in
    "$PWD"/sample_[0-9]*_storage_run)
      [ "$target" != "$PWD/sample_10000_storage_run" ] || exit 1
      rm -rf --one-file-system -- "$target"
      ;;
    *) echo "Refusing cleanup target: $target" >&2; exit 1 ;;
  esac
done

# Copy a deterministic random 10,000-song sample from ./audio/.
bash sample_10000_storage_run/copy_10000_from_downloaded.sh \
  --source-dir "$PWD/audio" \
  --n 10000 \
  --seed 42 \
  --workers 16

# Pick a free GPU first.
nvidia-smi
export CUDA_VISIBLE_DEVICES=<gpu_index>

# Run preprocessing, handcrafted features, pretrained embeddings, and report.
bash sample_10000_storage_run/run_dgx_storage_benchmark.sh
```

The DGX scripts write outputs and framework/model caches only under
`sample_10000_storage_run/`. The runner also verifies that MusicNN, MERT and
EnCodecMAE all load before processing, and verifies that all three models
successfully produced embeddings for all 10,000 songs before writing the final
storage report.

The PowerShell helper is only for local Windows use after a source directory is
explicitly provided:

```powershell
cd C:\Users\vpddk\Desktop\Me\Github\ML-song-recommendation-system
.\msd_deezer_workspace\sample_10000_storage_run\copy_10000_from_downloaded.ps1 `
  -SourceDir "C:\path\to\downloaded\audio"
```

Both copy helpers refuse to run if the source has fewer than 10,000 audio files
or if `audio_10000/` is not empty, unless the DGX Python helper is explicitly
called with `--force`.
