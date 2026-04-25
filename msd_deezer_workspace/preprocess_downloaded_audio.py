from __future__ import annotations

from audio_preprocessing import (
    DEFAULT_HANDCRAFTED_DIR,
    DEFAULT_PRETRAINED_DIR,
    DEFAULT_SOURCE_DIR,
    DualAudioPreprocessor,
)


def main() -> None:
    processor = DualAudioPreprocessor()
    processor.process_directory(
        source_dir=DEFAULT_SOURCE_DIR,
        handcrafted_dir=DEFAULT_HANDCRAFTED_DIR,
        pretrained_dir=DEFAULT_PRETRAINED_DIR,
    )


if __name__ == "__main__":
    main()
