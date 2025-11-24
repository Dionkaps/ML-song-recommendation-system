import requests
import os
import sys
import re
import time
import multiprocessing
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

DEEZE_AUDIO_FOLDER = "audio_files"
SONGS_DATA_CSV = "songs_data_with_genre.csv"
CHECKPOINT_FILE = "download_checkpoint_with_genre.json"
INPUT_CSV = "millionsong_dataset.csv"


def search_song(song_name):
    """Search for a song on Deezer by name"""
    try:
        search_url = f"https://api.deezer.com/search?q={song_name}"
        response = requests.get(search_url, timeout=10)

        if response.status_code != 200:
            print(f"Error searching for song: {response.status_code}")
            return None

        data = response.json()
        if data.get('total', 0) == 0:
            print(f"No songs found for '{song_name}'")
            return None

        # Return the first song result
        return data['data'][0]
    except requests.exceptions.Timeout:
        print(f"Timeout searching for '{song_name}'")
        return None
    except Exception as e:
        print(f"Error searching for '{song_name}': {str(e)}")
        return None


def download_preview(preview_url, output_file):
    """Download the audio preview from the given URL"""
    try:
        response = requests.get(preview_url, stream=True, timeout=15)

        if response.status_code != 200:
            print(f"Error downloading preview: {response.status_code}")
            return False

        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        return True
    except requests.exceptions.Timeout:
        print(f"Timeout downloading preview from {preview_url}")
        return False
    except Exception as e:
        print(f"Error downloading preview: {str(e)}")
        return False


def sanitize_filename(filename):
    # Remove or replace invalid characters for Windows filenames
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def download_and_save_song(song, artist_name=""):
    """Download and save a song preview, returning song metadata."""
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
        # Clean up partially downloaded file if it exists
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                print(f"Cleaned up failed download: {output_file}")
            except Exception as e:
                print(f"Warning: Could not remove failed file: {e}")
        return None


def process_song(song_data):
    """Process a single song from CSV data"""
    try:
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
        
        # Extract data from dict (for CSV) or parse string (for text file)
        if isinstance(song_data, dict):
            song_name = song_data.get('title', '').strip()
            artist_name = song_data.get('artist', '').strip()
            index = song_data.get('index', -1)
        else:
            # Fallback for text file format
            song_line = song_data
            song_name = ""
            artist_name = ""
            index = -1
            
            if '–' in song_line:
                parts = song_line.strip().split('–')
                if len(parts) == 2:
                    song_name = parts[0].strip()
                    artist_name = parts[1].strip()
            elif '-' in song_line:
                parts = song_line.strip().split('-')
                if len(parts) == 2:
                    song_name = parts[0].strip()
                    artist_name = parts[1].strip()
            
            # If no separator or invalid format, treat entire line as song name
            if not song_name:
                song_name = song_line.strip()

        #  Add genre validation BEFORE search/download
        genre = song_data.get('genre', '').strip() if isinstance(song_data, dict) else ''
        
        if not genre:
            print(f"[{index}] Skipping '{song_name}' - no genre in dataset")
            return None

        if not song_name:
            return None

        # Build search query
        if artist_name:
            search_query = f"{song_name} {artist_name}"
            print(f"[{index}] Searching for: {song_name} by {artist_name} (genre: {genre[:30]}...)")
        else:
            search_query = song_name
            print(f"[{index}] Searching for: {song_name} (genre: {genre[:30]}...)")

        # Search and download
        song = search_song(search_query)
        if not song:
            return None

        result = download_and_save_song(song, artist_name)
        if result:
            # Add genre to result (do NOT add 'index' - causes CSV save error)
            result['genre'] = genre
        return result

    except Exception as e:
        print(f"Error processing song: {str(e)}")
        return None


def load_checkpoint():
    """Load checkpoint data from file"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    return {'processed_indices': [], 'downloaded_songs': []}


def save_checkpoint(checkpoint_data):
    """Save checkpoint data to file"""
    try:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def create_download_statistics_graphs(total_songs, downloaded_count, failed_count, elapsed_time):
    """Create visualization graphs for download statistics"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "download_stats")
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    not_found = total_songs - downloaded_count - failed_count
    success_rate = (downloaded_count / total_songs * 100) if total_songs > 0 else 0
    failed_rate = (failed_count / total_songs * 100) if total_songs > 0 else 0
    not_found_rate = (not_found / total_songs * 100) if total_songs > 0 else 0
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Graph 1: Pie chart of download results
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sizes = [downloaded_count, failed_count, not_found]
    labels = [f'Downloaded\n({downloaded_count})', 
              f'Failed/No Preview\n({failed_count})', 
              f'Not Found\n({not_found})']
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    explode = (0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors, explode=explode, startangle=90,
                                        textprops={'fontsize': 11, 'weight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
    
    ax1.set_title(f'Download Results Summary\nTotal Songs: {total_songs}', 
                 fontsize=14, weight='bold', pad=20)
    
    fig1.tight_layout()
    pie_path = os.path.join(output_dir, f'download_pie_chart_{timestamp}.png')
    fig1.savefig(pie_path, dpi=300, bbox_inches='tight')
    print(f"✓ Pie chart saved: {pie_path}")
    plt.close(fig1)
    
    # Graph 2: Bar chart comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    categories = ['Downloaded', 'Failed/No Preview', 'Not Found']
    values = [downloaded_count, failed_count, not_found]
    percentages = [success_rate, failed_rate, not_found_rate]
    
    bars = ax2.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value, pct in zip(bars, values, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, weight='bold')
    
    ax2.set_ylabel('Number of Songs', fontsize=12, weight='bold')
    ax2.set_title('Download Statistics by Category', fontsize=14, weight='bold', pad=20)
    ax2.set_ylim(0, max(values) * 1.15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig2.tight_layout()
    bar_path = os.path.join(output_dir, f'download_bar_chart_{timestamp}.png')
    fig2.savefig(bar_path, dpi=300, bbox_inches='tight')
    print(f"✓ Bar chart saved: {bar_path}")
    plt.close(fig2)
    
    # Graph 3: Summary statistics text visualization
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.axis('off')
    
    # Calculate minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    
    # Songs per minute
    songs_per_minute = (total_songs / elapsed_time * 60) if elapsed_time > 0 else 0
    
    summary_text = f"""
    DOWNLOAD SESSION SUMMARY
    {'='*50}
    
    Total Songs Processed:        {total_songs}
    Successfully Downloaded:      {downloaded_count} ({success_rate:.1f}%)
    Failed/No Preview Available:  {failed_count} ({failed_rate:.1f}%)
    Not Found on Deezer:          {not_found} ({not_found_rate:.1f}%)
    
    {'='*50}
    
    Session Duration:             {time_str}
    Average Speed:                {songs_per_minute:.1f} songs/min
    
    Output Files:
    • CSV Results: songs_data.csv
    • Checkpoint: download_checkpoint.json
    • Audio Files: audio_files/ folder
    """
    
    ax3.text(0.5, 0.5, summary_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig3.tight_layout()
    summary_path = os.path.join(output_dir, f'download_summary_{timestamp}.png')
    fig3.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"✓ Summary statistics saved: {summary_path}")
    plt.close(fig3)
    
    print(f"\n📊 All graphs saved to: {output_dir}")
    return output_dir


def save_songs_to_csv(songs_data, filename):
    """Save the songs data to a CSV file"""
    # Filter out None values
    songs_data = [song for song in songs_data if song]

    if not songs_data:
        print("No song data to save.")
        return

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['title', 'artist', 'filename', 'genre']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for song in songs_data:
                writer.writerow(song)

        print(f"Successfully saved {len(songs_data)} songs to {filename}")
    except Exception as e:
        print(f"Error saving data to CSV: {str(e)}")


def main():
    start_time = time.time()
    
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, INPUT_CSV)
    
    # Check if Million Song CSV exists, fallback to names.txt
    if os.path.exists(csv_path):
        print(f"Loading songs from {INPUT_CSV}...")
        songs_list = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for idx, row in enumerate(reader):
                    row['index'] = idx
                    songs_list.append(row)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return
    else:
        # Fallback to old text file format
        file_path = "config/names.txt"
        print(f"CSV not found, falling back to {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"Error: Neither '{INPUT_CSV}' nor '{file_path}' found.")
            return

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        if not lines:
            print(f"Error: '{file_path}' is empty.")
            return
        
        # Filter out empty lines and convert to dict format
        songs_list = [{'title': line.strip(), 'artist': '', 'index': idx} 
                     for idx, line in enumerate(lines) if line.strip()]

    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_indices = set(checkpoint.get('processed_indices', []))
    
    # Filter out already processed songs
    songs_to_process = [song for song in songs_list 
                       if song.get('index', -1) not in processed_indices]
    
    total_count = len(songs_list)
    already_processed = len(processed_indices)
    remaining = len(songs_to_process)
    
    print(f"\nTotal songs: {total_count}")
    print(f"Already processed: {already_processed}")
    print(f"Remaining to process: {remaining}")
    
    if remaining == 0:
        print("All songs already processed!")
        return

    # Ensure the output folder exists
    os.makedirs(DEEZE_AUDIO_FOLDER, exist_ok=True)

    # Determine number of worker threads
    num_workers = min(32, remaining, multiprocessing.cpu_count() * 4)
    print(f"Starting parallel processing with {num_workers} workers...\n")

    songs_data = checkpoint.get('downloaded_songs', [])
    completed_in_session = 0
    failed_in_session = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_song = {executor.submit(
            process_song, song): song for song in songs_to_process}

        # Process results as they complete
        for future in as_completed(future_to_song):
            song_data = future_to_song[future]
            song_index = song_data.get('index', -1)
            
            try:
                result = future.result(timeout=30)
                
                # Mark as processed regardless of success
                processed_indices.add(song_index)
                
                if result:
                    songs_data.append(result)
                    completed_in_session += 1
                else:
                    failed_in_session += 1
                
                # Save checkpoint every 10 songs
                total_processed = completed_in_session + failed_in_session
                if total_processed % 10 == 0:
                    checkpoint['processed_indices'] = list(processed_indices)
                    checkpoint['downloaded_songs'] = songs_data
                    save_checkpoint(checkpoint)
                    print(f"\n--- Checkpoint saved: {total_processed}/{remaining} songs processed ({completed_in_session} success, {failed_in_session} failed) ---\n")
            except Exception as e:
                print(f"Error processing song at index {song_index}: {str(e)}")
                # Mark as processed even if there was an error
                processed_indices.add(song_index)
                failed_in_session += 1

    # Final checkpoint save
    checkpoint['processed_indices'] = list(processed_indices)
    checkpoint['downloaded_songs'] = songs_data
    save_checkpoint(checkpoint)

    # Count successful downloads
    success_count = len(songs_data)
    total_processed_in_session = completed_in_session + failed_in_session

    # Save song data to CSV
    save_songs_to_csv(songs_data, SONGS_DATA_CSV)

    elapsed = time.time() - start_time
    print(
        f"\nSession completed: {total_processed_in_session} songs processed")
    print(f"  - Successfully downloaded: {success_count}")
    print(f"  - Failed/Not found: {failed_in_session}")
    print(f"Total successful downloads so far: {success_count} of {total_count} songs")
    print(f"Session time: {elapsed:.2f} seconds")
    
    # Create visualization graphs
    print("\n" + "="*50)
    print("Generating download statistics graphs...")
    print("="*50)
    
    # Calculate total failed (including previous sessions)
    total_failed = len(processed_indices) - success_count
    create_download_statistics_graphs(total_count, success_count, total_failed, elapsed)
    
    if len(processed_indices) < total_count:
        print(f"\nNote: {total_count - len(processed_indices)} songs remaining. Run script again to continue.")
    else:
        print("\nAll songs processed! You can delete '{CHECKPOINT_FILE}' to start fresh.")
    print(f"Song data saved to {SONGS_DATA_CSV}")


if __name__ == "__main__":
    main()