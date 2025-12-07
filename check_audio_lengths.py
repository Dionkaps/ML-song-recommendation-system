"""
Script to check the duration of all audio files in the audio_files directory.
Verifies if all audio files are 29 seconds long.
"""

import os
from mutagen.mp3 import MP3
from mutagen import MutagenError

def get_audio_duration(file_path):
    """Get the duration of an audio file in seconds."""
    try:
        audio = MP3(file_path)
        return audio.info.length
    except MutagenError as e:
        # Fallback to pydub for problematic files
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(file_path)
            return len(audio) / 1000.0  # Convert ms to seconds
        except Exception as e2:
            print(f"Error reading {file_path}: mutagen: {e}, pydub: {e2}")
            return None

def check_audio_lengths(audio_dir, target_duration=29.0, tolerance=0.5):
    """
    Check all audio files in the directory and report their lengths.
    
    Args:
        audio_dir: Path to the directory containing audio files
        target_duration: Expected duration in seconds (default: 29.0)
        tolerance: Acceptable deviation from target duration in seconds (default: 0.5)
    """
    if not os.path.exists(audio_dir):
        print(f"Error: Directory '{audio_dir}' does not exist!")
        return
    
    # Get all mp3 files
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.mp3')]
    
    if not audio_files:
        print(f"No MP3 files found in '{audio_dir}'")
        return
    
    print(f"Checking {len(audio_files)} audio files...")
    print(f"Target duration: {target_duration} seconds (±{tolerance}s tolerance)")
    print("-" * 80)
    
    correct_length = []
    incorrect_length = []
    errors = []
    
    for filename in sorted(audio_files):
        file_path = os.path.join(audio_dir, filename)
        duration = get_audio_duration(file_path)
        
        if duration is None:
            errors.append(filename)
        elif abs(duration - target_duration) <= tolerance:
            correct_length.append((filename, duration))
        else:
            incorrect_length.append((filename, duration))
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files checked: {len(audio_files)}")
    print(f"Files with correct length (~{target_duration}s): {len(correct_length)}")
    print(f"Files with incorrect length: {len(incorrect_length)}")
    print(f"Files with errors: {len(errors)}")
    
    # Print files with incorrect length
    if incorrect_length:
        print("\n" + "-" * 80)
        print("FILES WITH INCORRECT LENGTH:")
        print("-" * 80)
        for filename, duration in sorted(incorrect_length, key=lambda x: x[1]):
            print(f"  {filename}: {duration:.2f}s")
    
    # Print files with errors
    if errors:
        print("\n" + "-" * 80)
        print("FILES WITH ERRORS:")
        print("-" * 80)
        for filename in errors:
            print(f"  {filename}")
    
    # Calculate statistics
    all_durations = [d for _, d in correct_length + incorrect_length]
    if all_durations:
        print("\n" + "-" * 80)
        print("STATISTICS:")
        print("-" * 80)
        print(f"  Minimum duration: {min(all_durations):.2f}s")
        print(f"  Maximum duration: {max(all_durations):.2f}s")
        print(f"  Average duration: {sum(all_durations)/len(all_durations):.2f}s")
    
    return {
        'total': len(audio_files),
        'correct': len(correct_length),
        'incorrect': len(incorrect_length),
        'errors': len(errors),
        'incorrect_files': incorrect_length,
        'error_files': errors
    }


if __name__ == "__main__":
    # Path to the audio files directory
    audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")
    
    # Run the check with 0 tolerance (exactly 29 seconds)
    results = check_audio_lengths(audio_dir, target_duration=29.0, tolerance=0.0)

