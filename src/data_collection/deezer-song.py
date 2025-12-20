import requests
import argparse
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
from pathlib import Path
from tqdm import tqdm

# Ensure we are running from project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

DEEZE_AUDIO_FOLDER = "audio_files"
UNIFIED_CSV = os.path.join("data", "songs.csv")  # Unified CSV with all song data
CHECKPOINT_FILE = "download_checkpoint_with_genre.json"
INPUT_CSV = os.path.join("data", "millionsong_dataset.csv")
SEARCH_CACHE_FILE = "deezer_search_cache.json"  # Cache successful searches

# Global search cache
_search_cache = {}


def load_search_cache():
    """Load cached search results from previous runs"""
    global _search_cache
    if os.path.exists(SEARCH_CACHE_FILE):
        try:
            with open(SEARCH_CACHE_FILE, 'r', encoding='utf-8') as f:
                _search_cache = json.load(f)
        except Exception:
            _search_cache = {}
    return _search_cache


def save_search_cache():
    """Save search cache to file for reuse in future runs"""
    try:
        with open(SEARCH_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(_search_cache, f, indent=2)
    except Exception:
        pass


def search_song(song_name, artist_name="", use_cache=True):
    """
    Search for a song on Deezer using song name + artist name.
    Uses caching to avoid redundant API calls.
    """
    # Check cache first
    cache_key = f"{song_name}|{artist_name}"
    if use_cache and cache_key in _search_cache:
        return _search_cache[cache_key]
    
    # Build search query: song + artist
    if artist_name:
        query = f"{song_name} {artist_name}"
    else:
        query = song_name
    
    try:
        search_url = f"https://api.deezer.com/search?q={query}"
        response = requests.get(search_url, timeout=10)

        if response.status_code != 200:
            if use_cache:
                _search_cache[cache_key] = None
            return None

        data = response.json()
        if data.get('total', 0) == 0:
            if use_cache:
                _search_cache[cache_key] = None
            return None

        # Return the first song result and cache it
        result = data['data'][0]
        if use_cache:
            _search_cache[cache_key] = result
        return result
        
    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        return None


def download_preview(preview_url, output_file):
    """Download the audio preview from the given URL"""
    try:
        response = requests.get(preview_url, stream=True, timeout=15)

        if response.status_code != 200:
            return False

        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        return True
    except requests.exceptions.Timeout:
        return False
    except Exception as e:
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
        return None

    # Ensure the output folder exists
    if not os.path.exists(DEEZE_AUDIO_FOLDER):
        os.makedirs(DEEZE_AUDIO_FOLDER)

    # Create a valid filename from the song title and artist
    raw_filename = f"{artist} - {title}"
    valid_filename = sanitize_filename(raw_filename)
    output_file = os.path.join(DEEZE_AUDIO_FOLDER, f"{valid_filename}.mp3")

    success = download_preview(preview_url, output_file)

    if success:
        return {"title": title, "artist": artist, "filename": valid_filename + ".mp3"}
    else:
        # Clean up partially downloaded file if it exists
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception:
                pass
        return None


def process_song(song_data):
    """Process a single song from CSV data"""
    try:
        # Add delay to avoid rate limiting (0.5s is safer for Deezer API)
        time.sleep(0.5)
        
        # Extract data from dict (for CSV) or parse string (for text file)
        if isinstance(song_data, dict):
            song_name = song_data.get('title', '').strip()
            artist_name = song_data.get('artist', '').strip()
            index = song_data.get('index', -1)
            msd_track_id = song_data.get('track_id', None)  # MSD track ID for matching
        else:
            # Fallback for text file format
            song_line = song_data
            song_name = ""
            artist_name = ""
            index = -1
            msd_track_id = None
            
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
        
        if not genre or genre.lower() == 'unknown':
            return None

        if not song_name:
            return None

        # Search and download using simple song + artist search
        song = search_song(song_name, artist_name, use_cache=True)
        if not song:
            return None

        result = download_and_save_song(song, artist_name)
        if result:
            # Store both MSD (original) and Deezer (downloaded) info for the unified CSV
            result['genre'] = genre
            result['msd_artist'] = artist_name  # Original MSD artist used for search
            result['msd_title'] = song_name     # Original MSD title used for search
            result['msd_track_id'] = msd_track_id  # MSD track ID for direct matching
            result['deezer_artist'] = result.pop('artist')  # Rename to clarify source
            result['deezer_title'] = result.pop('title')    # Rename to clarify source
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


def update_unified_csv(downloaded_songs):
    """
    Update the unified songs.csv with download results.
    
    This updates existing rows with Deezer info (artist, title, filename)
    rather than creating a separate CSV.
    """
    import pandas as pd
    
    if not downloaded_songs:
        print("No songs to update in unified CSV.")
        return
    
    if not os.path.exists(UNIFIED_CSV):
        print(f"Warning: Unified CSV not found at {UNIFIED_CSV}")
        print("Creating new unified CSV from download data...")
        # Fallback: create basic CSV if unified doesn't exist
        df = pd.DataFrame(downloaded_songs)
        df.columns = ['deezer_title', 'deezer_artist', 'filename', 'genre', 'msd_artist', 'msd_title', 'msd_track_id']
        df['has_audio'] = True
        df.to_csv(UNIFIED_CSV, index=False)
        return
    
    try:
        # Load existing unified CSV
        unified = pd.read_csv(UNIFIED_CSV)
        
        def normalize(s):
            if pd.isna(s):
                return ""
            return str(s).lower().strip()
        
        updated_count = 0
        for song in downloaded_songs:
            if song is None:
                continue
            
            # Match by MSD track_id if available, else by artist+title
            msd_track_id = song.get('msd_track_id')
            msd_artist = song.get('msd_artist', song.get('artist', ''))
            msd_title = song.get('msd_title', song.get('title', ''))
            
            if msd_track_id:
                mask = unified['msd_track_id'] == msd_track_id
            else:
                mask = (unified['msd_artist'].apply(normalize) == normalize(msd_artist)) & \
                       (unified['msd_title'].apply(normalize) == normalize(msd_title))
            
            if mask.any():
                idx = unified[mask].index[0]
                unified.loc[idx, 'deezer_artist'] = song.get('deezer_artist', song.get('artist'))
                unified.loc[idx, 'deezer_title'] = song.get('deezer_title', song.get('title'))
                unified.loc[idx, 'filename'] = song.get('filename')
                unified.loc[idx, 'has_audio'] = True
                updated_count += 1
        
        # Save updated unified CSV
        unified.to_csv(UNIFIED_CSV, index=False)
        print(f"Updated {updated_count} songs in unified CSV: {UNIFIED_CSV}")
        
    except Exception as e:
        print(f"Error updating unified CSV: {str(e)}")
        # Fallback to old method
        save_songs_to_csv_legacy(downloaded_songs, "data/songs_data_with_genre.csv")


def save_songs_to_csv_legacy(songs_data, filename):
    """Legacy method: Save the songs data to a separate CSV file"""
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
                # Map to legacy format
                row = {
                    'title': song.get('deezer_title', song.get('title')),
                    'artist': song.get('deezer_artist', song.get('artist')),
                    'filename': song.get('filename'),
                    'genre': song.get('genre')
                }
                writer.writerow(row)

        print(f"Successfully saved {len(songs_data)} songs to {filename}")
    except Exception as e:
        print(f"Error saving data to CSV: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Download song previews from Deezer")
    parser.add_argument("--limit", type=int, help="Limit number of songs to download this session")
    args = parser.parse_args()

    start_time = time.time()
    
    # Load search cache from previous runs
    load_search_cache()
    if _search_cache:
        print(f"📦 Loaded {len(_search_cache)} cached search results\n")
    
    # Use path relative to project root (since we chdir'd there)
    csv_path = INPUT_CSV
    
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

    # Apply limit if specified
    if args.limit:
        print(f"⚠️  Limit applied: Processing only first {args.limit} available songs")
        songs_to_process = songs_to_process[:args.limit]
    
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

    # Determine number of worker threads (reduced to 8 to avoid API rate limits)
    num_workers = min(8, remaining)
    print(f"Starting parallel processing with {num_workers} workers...\n")

    songs_data = checkpoint.get('downloaded_songs', [])
    total_completed = 0
    total_failed = 0
    
    # Track which songs failed in the current run (separate from checkpoint)
    failed_indices_this_run = set()
    
    # Run download attempts - retry as long as we get successful downloads
    attempt = 1
    while True:
        if attempt > 1:
            # On retry attempts, only process songs that failed in this run
            print(f"\n{'='*60}")
            print(f"RETRY ATTEMPT {attempt - 1} - Re-processing failed songs")
            print(f"{'='*60}\n")
            
            # Filter to only songs that failed (not in processed_indices but in failed_indices_this_run)
            songs_to_process = [song for song in songs_list 
                               if song.get('index', -1) in failed_indices_this_run]
            
            if not songs_to_process:
                print("No more songs to retry!")
                break
            
            print(f"Retrying {len(songs_to_process)} previously failed songs...\n")
        
        completed_in_attempt = 0
        failed_in_attempt = 0
        failed_indices_this_attempt = set()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_song = {executor.submit(
                process_song, song): song for song in songs_to_process}

            # Process results as they complete with progress bar
            pbar = tqdm(as_completed(future_to_song), total=len(future_to_song), 
                        desc=f"Downloading (attempt {attempt})", unit="song")
            for future in pbar:
                song_data = future_to_song[future]
                song_index = song_data.get('index', -1)
                
                try:
                    result = future.result(timeout=30)
                    
                    if result:
                        # Success - mark as processed and remove from failed set
                        processed_indices.add(song_index)
                        failed_indices_this_run.discard(song_index)
                        songs_data.append(result)
                        completed_in_attempt += 1
                        total_completed += 1
                    else:
                        # Failed - track for retry but don't mark as processed
                        failed_indices_this_attempt.add(song_index)
                        failed_in_attempt += 1
                    
                    # Update progress bar description
                    pbar.set_postfix(success=completed_in_attempt, failed=failed_in_attempt)
                    
                    # Save checkpoint every 10 successful downloads
                    if total_completed % 10 == 0 and total_completed > 0:
                        checkpoint['processed_indices'] = list(processed_indices)
                        checkpoint['downloaded_songs'] = songs_data
                        save_checkpoint(checkpoint)
                except Exception as e:
                    # Exception - track for retry but don't mark as processed
                    failed_indices_this_attempt.add(song_index)
                    failed_in_attempt += 1
                    pbar.set_postfix(success=completed_in_attempt, failed=failed_in_attempt)
        
        # Update total failed count and failed indices for next retry
        total_failed = len(failed_indices_this_attempt)
        failed_indices_this_run = failed_indices_this_attempt
        
        print(f"\nAttempt {attempt} complete: {completed_in_attempt} success, {failed_in_attempt} failed")
        
        # Stop if no successes in this attempt (no point retrying)
        if completed_in_attempt == 0:
            print("No successful downloads in this attempt - stopping retries.")
            break
        
        # Stop if no failures to retry
        if not failed_indices_this_run:
            print("All songs downloaded successfully!")
            break
        
        # Continue to next retry with exponential backoff
        attempt += 1
        backoff_time = 5 * (2 ** (attempt - 2))  # 5s, 10s, 20s, 40s...
        backoff_time = min(backoff_time, 60)  # Cap at 60 seconds
        print(f"\nWaiting {backoff_time} seconds before retry (exponential backoff)...\n")
        time.sleep(backoff_time)

    # Final checkpoint save
    checkpoint['processed_indices'] = list(processed_indices)
    checkpoint['downloaded_songs'] = songs_data
    save_checkpoint(checkpoint)

    # Count successful downloads
    success_count = len(songs_data)

    # Update unified CSV with download results
    update_unified_csv(songs_data)

    elapsed = time.time() - start_time
    print(
        f"\nSession completed: {total_completed} downloaded, {total_failed} still failed")
    print(f"Total successful downloads so far: {success_count} of {total_count} songs")
    print(f"Session time: {elapsed:.2f} seconds")
    
    # Save search cache for future runs
    save_search_cache()
    print(f"\n💾 Saved {len(_search_cache)} search results to cache for next run")
    
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
        print(f"\nAll songs processed! You can delete '{CHECKPOINT_FILE}' to start fresh.")
    print(f"Song data saved to {UNIFIED_CSV}")


if __name__ == "__main__":
    main()