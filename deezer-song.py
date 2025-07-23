import requests
import os
import sys
import re
import time
import multiprocessing
import csv
from concurrent.futures import ThreadPoolExecutor

DEEZE_AUDIO_FOLDER = "audio_files"
SONGS_DATA_CSV = "songs_data.csv"


def search_song(song_name):
    """Search for a song on Deezer by name"""
    search_url = f"https://api.deezer.com/search?q={song_name}"
    response = requests.get(search_url)

    if response.status_code != 200:
        print(f"Error searching for song: {response.status_code}")
        return None

    data = response.json()
    if data.get('total', 0) == 0:
        print(f"No songs found for '{song_name}'")
        return None

    # Return the first song result
    return data['data'][0]


def download_preview(preview_url, output_file):
    """Download the audio preview from the given URL"""
    response = requests.get(preview_url, stream=True)

    if response.status_code != 200:
        print(f"Error downloading preview: {response.status_code}")
        return False

    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    return True


def sanitize_filename(filename):
    # Remove or replace invalid characters for Windows filenames
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def process_song(song_line):
    """Process a single song line from the file"""
    try:
        if '–' in song_line:
            parts = song_line.strip().split('–')
        elif '-' in song_line:
            parts = song_line.strip().split('-')
        else:
            # If there's no separator, assume the entire line is a song name
            print(
                f"No artist specified for song: {song_line}. Searching by song name only.")
            song_name = song_line.strip()
            artist_name = ""
            search_query = song_name

            print(f"Searching for: {song_name}")

            song = search_song(search_query)
            if not song:
                return None

            # Rest of the function continues here with the song object
            title = song.get('title', 'Unknown')
            artist = song.get('artist', {}).get('name', 'Unknown Artist')
            preview_url = song.get('preview', None)

            if not preview_url:
                print(f"No preview available for {title} by {artist}")
                return None

            print(f"Found: {title} by {artist}")

            # Ensure the output folder exists
            if not os.path.exists(DEEZE_AUDIO_FOLDER):
                os.makedirs(DEEZE_AUDIO_FOLDER)

            # Create a valid filename from the song title and artist
            raw_filename = f"{artist} - {title}"
            valid_filename = sanitize_filename(raw_filename)
            output_file = os.path.join(
                DEEZE_AUDIO_FOLDER, f"{valid_filename}.mp3")

            print(f"Downloading preview to {output_file}...")
            success = download_preview(preview_url, output_file)

            if success:
                print(f"Download complete! Saved to {output_file}")
                return {"title": title, "artist": artist, "filename": valid_filename + ".mp3"}
            else:
                print("Download failed")
                return None

        if len(parts) != 2:
            print(f"Invalid format in line: {song_line}")
            return None

        # Original code for when there is artist information
        song_name = parts[0].strip()
        artist_name = parts[1].strip()
        search_query = f"{song_name} {artist_name}"

        print(f"Searching for: {song_name} by {artist_name}")

        song = search_song(search_query)
        if not song:
            return None

        # Display song info
        title = song.get('title', 'Unknown')
        artist = song.get('artist', {}).get('name', 'Unknown Artist')
        preview_url = song.get('preview', None)

        if not preview_url:
            print(f"No preview available for {title} by {artist}")
            return None

        print(f"Found: {title} by {artist}")

        # Ensure the output folder exists
        if not os.path.exists(DEEZE_AUDIO_FOLDER):
            os.makedirs(DEEZE_AUDIO_FOLDER)

        # Create a valid filename from the song title and artist
        raw_filename = f"{artist} - {title}"
        valid_filename = sanitize_filename(raw_filename)
        output_file = os.path.join(DEEZE_AUDIO_FOLDER, f"{valid_filename}.mp3")

        print(f"Downloading preview to {output_file}...")
        success = download_preview(preview_url, output_file)

        if success:
            print(f"Download complete! Saved to {output_file}")
            return {"title": title, "artist": artist, "filename": valid_filename + ".mp3"}
        else:
            print("Download failed")
            return None
    except Exception as e:
        print(f"Error processing song '{song_line}': {str(e)}")
        return None


def save_songs_to_csv(songs_data, filename):
    """Save the songs data to a CSV file"""
    # Filter out None values
    songs_data = [song for song in songs_data if song]

    if not songs_data:
        print("No song data to save.")
        return

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['title', 'artist', 'filename']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for song in songs_data:
                writer.writerow(song)

        print(f"Successfully saved {len(songs_data)} songs to {filename}")
    except Exception as e:
        print(f"Error saving data to CSV: {str(e)}")


def main():
    start_time = time.time()
    file_path = "names.txt"

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    if not lines:
        print(f"Error: '{file_path}' is empty.")
        return

    # Ensure the output folder exists
    os.makedirs(DEEZE_AUDIO_FOLDER, exist_ok=True)

    # Filter out empty lines
    lines = [line.strip() for line in lines if line.strip()]
    total_count = len(lines)

    print(f"Found {total_count} songs to process")

    # Determine number of worker threads (I/O bound, so we can use more threads than CPUs)
    # Use ThreadPoolExecutor since this is I/O bound work
    num_workers = min(32, total_count, multiprocessing.cpu_count() * 4)
    print(f"Starting parallel processing with {num_workers} workers...")

    songs_data = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_song = {executor.submit(
            process_song, line): line for line in lines}

        # Process results as they complete
        for future in future_to_song:
            songs_data.append(future.result())

    # Count successful downloads
    success_count = sum(1 for result in songs_data if result)

    # Save song data to CSV
    save_songs_to_csv(songs_data, SONGS_DATA_CSV)

    elapsed = time.time() - start_time
    print(
        f"\nCompleted: {success_count} of {total_count} songs downloaded successfully.")
    print(f"Total elapsed time: {elapsed:.2f} seconds")
    print(f"Song data saved to {SONGS_DATA_CSV}")


if __name__ == "__main__":
    main()
