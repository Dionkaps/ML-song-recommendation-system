import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from aubio import source, tempo
from numpy import median, diff
from yt_dlp import YoutubeDL

################################################################################
#                               DOWNLOAD SECTION                               #
################################################################################


def download_videos(video_urls, ydl_opts):
    with YoutubeDL(ydl_opts) as ydl:
        for url in video_urls:
            success = False
            for attempt in range(3):
                try:
                    print(f"Attempting to download: {
                          url} (Attempt {attempt+1})")
                    info = ydl.extract_info(url, download=True)
                    title = info.get('title', 'Unknown Title')
                    print(f"Successfully downloaded: {title}")
                    success = True
                    break
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
                    if attempt < 2:
                        print("Retrying in 10 seconds...")
                        time.sleep(10)
                    else:
                        print("Maximum retries reached. Skipping to the next video.")
            if not success:
                print(f"Failed to download {url}")


def download_from_links(links_file='links.txt'):
    # Create output directory (if it doesn't exist)
    os.makedirs('audio_files', exist_ok=True)

    # Read links from links_file
    try:
        with open(links_file, 'r', encoding='utf-8') as f:
            playlist_links = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"'{links_file}' not found.")
        return

    for video_url in playlist_links:
        # Options for yt_dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'keepvideo': False,
            'outtmpl': 'audio_files/%(title)s.%(ext)s',
            'ignoreerrors': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }

        print(f"\nProcessing URL/Playlist: {video_url}")
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                # If it's a playlist
                if 'entries' in info:
                    playlist_title = info.get('title', 'Unknown Playlist')
                    print(f"Detected playlist: {playlist_title}")
                    entries = info['entries']

                    # Download in batches of 10 to avoid hitting limits
                    for i in range(0, len(entries), 10):
                        batch = entries[i:i + 10]
                        urls = [entry['webpage_url']
                                for entry in batch if entry]
                        download_videos(urls, ydl_opts)
                else:
                    # Single video
                    title = info.get('title', 'Unknown Title')
                    print(f"Single video detected: {title}")
                    download_videos([info['webpage_url']], ydl_opts)

        except Exception as e:
            print(f"An error occurred while processing {video_url}: {e}")


################################################################################
#                         FEATURE EXTRACTION SECTION                           #
################################################################################

def extract_mfcc(y, sr, n_mfcc=13):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


def extract_melspectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                                     hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def extract_spectral_centroid(y, sr, hop_length=512, n_fft=2048):
    return librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)


def extract_spectral_flatness(y, hop_length=512, n_fft=2048):
    return librosa.feature.spectral_flatness(y=y, hop_length=hop_length, n_fft=n_fft)


def extract_zero_crossing_rate(y, hop_length=512):
    return librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)


def save_feature(feature, output_file):
    np.save(output_file, feature)


def run_feature_extraction(audio_dir='audio_files', results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)

    # Find all .wav files in audio_dir
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    if not wav_files:
        print(f"No .wav files found in {audio_dir}.")
        return

    # Parameters for feature extraction
    n_mfcc = 13
    n_fft = 2048
    hop_length = 512
    n_mels = 128

    for wav_path in wav_files:
        base_filename = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\nExtracting features from: {base_filename}.wav")

        output_mfcc_path = os.path.join(
            results_dir, f"{base_filename}_mfcc.npy")
        output_melspectrogram_path = os.path.join(
            results_dir, f"{base_filename}_melspectrogram.npy")
        output_spectral_centroid_path = os.path.join(
            results_dir, f"{base_filename}_spectral_centroid.npy")
        output_spectral_flatness_path = os.path.join(
            results_dir, f"{base_filename}_spectral_flatness.npy")
        output_zero_crossing_rate_path = os.path.join(
            results_dir, f"{base_filename}_zero_crossing_rate.npy")

        try:
            print(f"Loading audio file: {wav_path}")
            y, sr = librosa.load(wav_path, sr=None)
            print(f"Audio loaded successfully with sampling rate = {sr} Hz")

            # Extract features
            mfccs = extract_mfcc(y, sr, n_mfcc=n_mfcc)
            save_feature(mfccs, output_mfcc_path)

            mel_spectrogram = extract_melspectrogram(y, sr, n_fft=n_fft,
                                                     hop_length=hop_length, n_mels=n_mels)
            save_feature(mel_spectrogram, output_melspectrogram_path)

            spectral_centroid = extract_spectral_centroid(
                y, sr, hop_length=hop_length, n_fft=n_fft)
            save_feature(spectral_centroid, output_spectral_centroid_path)

            spectral_flatness = extract_spectral_flatness(
                y, hop_length=hop_length, n_fft=n_fft)
            save_feature(spectral_flatness, output_spectral_flatness_path)

            zcr = extract_zero_crossing_rate(y, hop_length=hop_length)
            save_feature(zcr, output_zero_crossing_rate_path)

            print(f"Features extracted for '{base_filename}.wav'.")
        except Exception as e:
            print(f"An error occurred while processing '{
                  base_filename}.wav': {e}")


################################################################################
#                            BPM EXTRACTION SECTION                            #
################################################################################

def get_file_bpm(file_path, params=None):
    if params is None:
        params = {}

    # Defaults
    samplerate, win_s, hop_s = 44100, 1024, 512

    # Parameters based on 'mode'
    if 'mode' in params:
        if params['mode'] == 'super-fast':
            samplerate, win_s, hop_s = 4000, 128, 64
        elif params['mode'] == 'fast':
            samplerate, win_s, hop_s = 8000, 512, 128
        elif params['mode'] == 'default':
            pass
        else:
            raise ValueError(f"Unknown mode {params['mode']}")

    samplerate = params.get('samplerate', samplerate)
    win_s = params.get('win_s', win_s)
    hop_s = params.get('hop_s', hop_s)

    try:
        s = source(file_path, samplerate, hop_s)
        samplerate = s.samplerate
        o = tempo("specdiff", win_s, hop_s, samplerate)

        beats = []
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

        if len(beats) > 1:
            bpms = 60.0 / diff(beats)
            return median(bpms)
        else:
            print(f"Not enough beats found in {file_path}.")
            return 0.0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0.0


def run_bpm_extraction(audio_dir='audio_files', mode='default', samplerate=48000):
    supported_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    if not os.path.isdir(audio_dir):
        print(f"Directory not found: {audio_dir}")
        return

    files_in_dir = os.listdir(audio_dir)
    audio_files = [f for f in files_in_dir if f.lower().endswith(
        supported_extensions)]

    if not audio_files:
        print(f"No audio files found in '{audio_dir}'.")
        return

    print("\nBPM Extraction Results:")
    for audio_file in audio_files:
        file_path = os.path.join(audio_dir, audio_file)
        bpm = get_file_bpm(file_path, params={
                           'mode': mode, 'samplerate': samplerate})
        if bpm > 0:
            print(f"  {audio_file}: BPM = {bpm:.2f}")
        else:
            print(f"  {audio_file}: BPM could not be determined.")


################################################################################
#                                PLOTTING SECTION                              #
################################################################################

def plot_feature(feature, y, sr, hop_length,
                 feature_type='mfcc',
                 n_fft=2048,
                 output_dir=None,
                 base_filename=None):
    plt.figure(figsize=(10, 4))
    song_name = base_filename if base_filename else "Unknown Song"

    # Plot logic
    if feature_type == 'mfcc':
        librosa.display.specshow(
            feature, x_axis='time', sr=sr, hop_length=hop_length)
        plt.colorbar()
        plt.title(f'MFCCs for "{song_name}"')

    elif feature_type == 'melspectrogram':
        librosa.display.specshow(feature, x_axis='time', y_axis='mel',
                                 sr=sr, hop_length=hop_length, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-Spectrogram (dB) for "{song_name}"')

    elif feature_type == 'spectral_centroid':
        times = np.arange(len(y)) / sr
        plt.plot(times, y, alpha=0.4, label='Waveform')
        frames = range(len(feature[0]))
        t_centroid = librosa.frames_to_time(
            frames, sr=sr, hop_length=hop_length)
        centroid_hz = feature[0]
        plt.plot(t_centroid, centroid_hz, color='r', label='Spectral Centroid')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude / Frequency (Hz)')
        plt.title(f'Spectral Centroid for "{song_name}"')
        plt.legend()

    elif feature_type == 'spectral_flatness':
        frames = range(len(feature[0]))
        t_flatness = librosa.frames_to_time(
            frames, sr=sr, hop_length=hop_length)
        plt.plot(t_flatness, feature[0], color='g', label='Spectral Flatness')
        plt.xlabel('Time (s)')
        plt.ylabel('Flatness')
        plt.title(f'Spectral Flatness for "{song_name}"')
        plt.legend()

    elif feature_type == 'zero_crossing_rate':
        frames = range(len(feature[0]))
        t_zcr = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        plt.plot(t_zcr, feature[0], color='m', label='Zero-Crossing Rate')
        plt.xlabel('Time (s)')
        plt.ylabel('Rate')
        plt.title(f'Zero-Crossing Rate for "{song_name}"')
        plt.legend()

    else:
        raise ValueError("Invalid feature_type for plotting.")

    plt.tight_layout()
    if output_dir and base_filename:
        plot_filename = f"{base_filename}_{feature_type}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        plt.close()
    else:
        plt.show()


def run_plotting(audio_dir='audio_files', results_dir='results'):
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    if not wav_files:
        print("No .wav files found in the audio_files directory.")
        return

    for wav_path in wav_files:
        base_filename = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\nGenerating plots for: {base_filename}.wav")

        # Paths to feature .npy
        feature_files = {
            'mfcc': os.path.join(results_dir, f"{base_filename}_mfcc.npy"),
            'melspectrogram': os.path.join(results_dir, f"{base_filename}_melspectrogram.npy"),
            'spectral_centroid': os.path.join(results_dir, f"{base_filename}_spectral_centroid.npy"),
            'spectral_flatness': os.path.join(results_dir, f"{base_filename}_spectral_flatness.npy"),
            'zero_crossing_rate': os.path.join(results_dir, f"{base_filename}_zero_crossing_rate.npy")
        }

        # Check for missing feature files
        missing_features = [
            ft for ft, path in feature_files.items() if not os.path.isfile(path)]
        if missing_features:
            print(
                f"Missing feature files for '{base_filename}.wav': "
                f"{', '.join(missing_features)}. Skipping."
            )
            continue

        try:
            y, sr = librosa.load(wav_path, sr=None)
            print(f"Loaded '{base_filename}.wav' with sampling rate = {sr} Hz")

            mfccs = np.load(feature_files['mfcc'])
            mel_spectrogram = np.load(feature_files['melspectrogram'])
            spectral_centroid = np.load(feature_files['spectral_centroid'])
            spectral_flatness = np.load(feature_files['spectral_flatness'])
            zero_crossing_rate = np.load(feature_files['zero_crossing_rate'])

            hop_length = 512
            n_fft = 2048

            plot_feature(mfccs, y=y, sr=sr, hop_length=hop_length,
                         feature_type='mfcc', output_dir=results_dir,
                         base_filename=base_filename)

            plot_feature(mel_spectrogram, y=y, sr=sr, hop_length=hop_length,
                         feature_type='melspectrogram', output_dir=results_dir,
                         base_filename=base_filename)

            plot_feature(spectral_centroid, y=y, sr=sr, hop_length=hop_length,
                         feature_type='spectral_centroid', n_fft=n_fft,
                         output_dir=results_dir, base_filename=base_filename)

            plot_feature(spectral_flatness, y=y, sr=sr, hop_length=hop_length,
                         feature_type='spectral_flatness', output_dir=results_dir,
                         base_filename=base_filename)

            plot_feature(zero_crossing_rate, y=y, sr=sr, hop_length=hop_length,
                         feature_type='zero_crossing_rate', output_dir=results_dir,
                         base_filename=base_filename)

            print(f"Plots saved for '{base_filename}.wav'.")
        except Exception as e:
            print(f"Error plotting features for '{base_filename}.wav': {e}")


################################################################################
#                               (MAIN FUNCTION)                                #
################################################################################

def run_full_pipeline():
    print("\n======= STEP 1: DOWNLOAD AUDIO FROM YOUTUBE =======")
    download_from_links('links.txt')

    print("\n======= STEP 2: EXTRACT AUDIO FEATURES =======")
    run_feature_extraction(audio_dir='audio_files', results_dir='results')

    print("\n======= STEP 3: CALCULATE BPM =======")
    run_bpm_extraction(audio_dir='audio_files',
                       mode='default', samplerate=48000)

    print("\n======= STEP 4: PLOT FEATURES =======")
    run_plotting(audio_dir='audio_files', results_dir='results')

    print("\n======= PIPELINE COMPLETE! =======")


if __name__ == "__main__":
    run_full_pipeline()
