from __future__ import annotations

from audio_preprocessing import (
    DEFAULT_HANDCRAFTED_DIR,
    DEFAULT_PRETRAINED_DIR,
    DEFAULT_SOURCE_DIR,
    DualAudioPreprocessor,
)


DEFAULT_WORKERS = 16


def main() -> None:
    processor = DualAudioPreprocessor()
    processor.process_directory(
        source_dir=DEFAULT_SOURCE_DIR,
        handcrafted_dir=DEFAULT_HANDCRAFTED_DIR,
        pretrained_dir=DEFAULT_PRETRAINED_DIR,
        max_workers=DEFAULT_WORKERS,
    )


if __name__ == "__main__":
    main()
