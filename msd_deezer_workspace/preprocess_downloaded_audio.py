from __future__ import annotations

from audio_preprocessing.processor import AudioPreprocessor, DEFAULT_AUDIO_DIR


def main() -> None:
    processor = AudioPreprocessor()
    processor.process_directory(str(DEFAULT_AUDIO_DIR))


if __name__ == "__main__":
    main()
