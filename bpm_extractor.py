import os
from aubio import source, tempo
from numpy import median, diff

def get_file_bpm(file_path, params=None):
    """Calculate the beats per minute (BPM) of a given audio file."""
    if params is None:
        params = {}

    # Default parameters
    samplerate, win_s, hop_s = 44100, 1024, 512

    # Adjust parameters based on mode
    if 'mode' in params:
        if params['mode'] == 'super-fast':
            samplerate, win_s, hop_s = 4000, 128, 64
        elif params['mode'] == 'fast':
            samplerate, win_s, hop_s = 8000, 512, 128
        elif params['mode'] == 'default':
            pass
        else:
            raise ValueError(f"Unknown mode {params['mode']}")

    # Override with any manual settings
    samplerate = params.get('samplerate', samplerate)
    win_s = params.get('win_s', win_s)
    hop_s = params.get('hop_s', hop_s)

    try:
        # Create source and tempo objects
        s = source(file_path, samplerate, hop_s)
        samplerate = s.samplerate
        o = tempo("specdiff", win_s, hop_s, samplerate)

        # List to store beat timestamps
        beats = []

        # Read frames from the audio file
        total_frames = 0
        while True:
            samples, read = s()
            is_beat = o(samples)
            if is_beat:
                this_beat = o.get_last_s()
                beats.append(this_beat)
            total_frames += read
            if read < hop_s:
                break

        # Calculate BPM
        if len(beats) > 1:
            bpms = 60.0 / diff(beats)
            return median(bpms)
        else:
            print(f"Not enough beats found in {file_path}")
            return 0.0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0.0

if __name__ == '__main__':
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the path to 'audio_files' directory
    audio_dir = os.path.join(script_dir, 'audio_files')

    # Check if the 'audio_files' directory exists
    if not os.path.isdir(audio_dir):
        print(f"Directory 'audio_files' not found in the path: {script_dir}")
    else:
        # Supported audio file extensions
        supported_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

        # List all files in the audio directory
        audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(supported_extensions)]

        if not audio_files:
            print(f"No audio files found in the directory: {audio_dir}")
        else:
            print(f"Found {len(audio_files)} audio file(s) in '{audio_dir}':\n")
            for audio_file in audio_files:
                file_path = os.path.join(audio_dir, audio_file)
                bpm = get_file_bpm(file_path, params={'mode': 'default', 'samplerate': 48000})
                if bpm > 0:
                    print(f"{audio_file}: BPM = {bpm:.2f}")
                else:
                    print(f"{audio_file}: BPM could not be determined.")
