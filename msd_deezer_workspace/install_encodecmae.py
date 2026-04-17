"""
Installer for the EnCodecMAE pretrained-embedding model.

Why this script exists:
  The upstream `encodecmae` package's setup.cfg declares
  `packages = encodecmae` (non-recursive), so `pip install` silently omits
  its five subpackages -- `models`, `configs`, `tasks`, `heareval_model`,
  `scripts`. Every import of `encodecmae.models` then explodes with
  `ModuleNotFoundError`. Upstream has not fixed this as of April 2026.

What this script does (idempotent -- safe to re-run):
  1. Pip-installs the package from GitHub if it's not already present.
  2. Clones the repo into a temp dir.
  3. Copies the missing subpackages into the installed location.
  4. Verifies `from encodecmae.models import EncodecMAE` now works.

Cross-platform: runs the same on the laptop (Windows) and DGX (Linux).

Usage:
  python msd_deezer_workspace/install_encodecmae.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_URL = "https://github.com/habla-liaa/encodecmae.git"
SUBPACKAGES = ("models", "configs", "tasks", "heareval_model", "scripts")

# Runtime deps that encodecmae imports at package-init / load_model time.
# We install these explicitly because the initial `pip install --no-deps`
# of encodecmae itself skips them, and re-runs of this installer won't
# re-trigger pip's dep resolver. Safe to pip-install repeatedly -- pip
# skips packages that are already at a satisfying version.
RUNTIME_DEPS = (
    "huggingface_hub",
    "gin-config",
    "encodec",
    "einops",
)


def run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def locate_installed_pkg() -> Path | None:
    """Return the encodecmae install dir, or None if not installed.

    Uses `pip show` rather than `import encodecmae`, because a broken
    install (missing subpackages) raises ImportError on import but is
    still recorded by pip.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "encodecmae"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None

    location: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("Location:"):
            location = line.split(":", 1)[1].strip()
            break
    if not location:
        return None

    pkg_dir = Path(location) / "encodecmae"
    return pkg_dir if pkg_dir.is_dir() else None


def pip_install_if_missing() -> Path:
    """Pip-install encodecmae from GitHub if not importable. Return its install dir."""
    pkg_dir = locate_installed_pkg()
    if pkg_dir is None:
        print("encodecmae not installed -- installing from GitHub...")
        run([sys.executable, "-m", "pip", "install", "--no-deps",
             f"git+{REPO_URL}"])
        pkg_dir = locate_installed_pkg()
        if pkg_dir is None:
            print("FAIL -- pip install succeeded but encodecmae still not importable")
            sys.exit(1)
    return pkg_dir


def subpackages_complete(pkg_dir: Path) -> bool:
    """Return True if every required subpackage directory exists."""
    return all((pkg_dir / sub).is_dir() for sub in SUBPACKAGES)


def patch_missing_subpackages(pkg_dir: Path) -> None:
    """Clone the repo and copy the five subpackages that pip omitted."""
    print(f"Patching subpackages into {pkg_dir}...")
    with tempfile.TemporaryDirectory(prefix="encodecmae_src_") as tmp:
        tmp_path = Path(tmp)
        run(["git", "clone", "--depth", "1", REPO_URL, str(tmp_path / "src")])
        src_root = tmp_path / "src" / "encodecmae"

        for sub in SUBPACKAGES:
            src = src_root / sub
            dst = pkg_dir / sub
            if not src.is_dir():
                print(f"  WARNING: {sub}/ not in upstream repo -- skipping")
                continue
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  copied {sub}/")


def verify() -> None:
    """Confirm `encodecmae.models` imports cleanly in a fresh interpreter.

    Runs in a subprocess so we don't have to worry about cached imports
    or gin's configurable-registry (it rejects double-registration).
    """
    result = subprocess.run(
        [sys.executable, "-c", "from encodecmae.models import EncodecMAE; print('import ok')"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("\nFAIL -- encodecmae.models still not importable:")
        print(result.stderr)
        sys.exit(1)
    print(f"\nOK -- {result.stdout.strip()}")


def ensure_runtime_deps() -> None:
    """Install encodecmae's runtime deps that `--no-deps` skipped.

    Pip is a no-op when a package is already installed at a satisfying
    version, so this is cheap on re-runs.
    """
    print("Ensuring encodecmae runtime dependencies are installed...")
    run([sys.executable, "-m", "pip", "install", *RUNTIME_DEPS])


def main() -> int:
    pkg_dir = pip_install_if_missing()
    print(f"encodecmae installed at: {pkg_dir}")

    if subpackages_complete(pkg_dir):
        print("All subpackages already present -- nothing to patch.")
    else:
        patch_missing_subpackages(pkg_dir)

    ensure_runtime_deps()
    verify()
    return 0


if __name__ == "__main__":
    sys.exit(main())
