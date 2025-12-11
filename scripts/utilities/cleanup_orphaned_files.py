"""
Cleanup orphaned MP3 files that don't have entries in unified songs.csv
"""
import os
import csv
import sys
from pathlib import Path

# Ensure we are running from project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

AUDIO_FOLDER = "audio_files"
# Try unified CSV first, then legacy
CSV_FILE = os.path.join("data", "songs.csv")
if not os.path.exists(CSV_FILE):
    CSV_FILE = os.path.join("data", "songs_data_with_genre.csv")

def cleanup_orphaned_files(auto_confirm=False):
    """Remove MP3 files that aren't in the CSV"""
    
    # Load filenames from CSV
    csv_filenames = set()
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('filename', '').strip()
                if filename:
                    csv_filenames.add(filename)
        print(f"Found {len(csv_filenames)} songs in {CSV_FILE}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Get all MP3 files in audio folder
    mp3_files = set()
    if os.path.exists(AUDIO_FOLDER):
        for filename in os.listdir(AUDIO_FOLDER):
            if filename.lower().endswith('.mp3'):
                mp3_files.add(filename)
        print(f"Found {len(mp3_files)} MP3 files in {AUDIO_FOLDER}/")
    else:
        print(f"Error: {AUDIO_FOLDER}/ folder not found")
        return
    
    # Find orphaned files (in folder but not in CSV)
    orphaned = mp3_files - csv_filenames
    
    if not orphaned:
        print("\n✓ No orphaned files found! All MP3s have CSV entries.")
        return
    
    print(f"\n⚠️ Found {len(orphaned)} orphaned MP3 files without CSV entries:")
    print("=" * 60)
    
    # Show sample
    print(f"\nFirst 10 orphaned files:")
    for i, filename in enumerate(list(orphaned)[:10]):
        print(f"  - {filename}")
    if len(orphaned) > 10:
        print(f"  ... and {len(orphaned) - 10} more")
    
    print("\n" + "=" * 60)
    
    # Ask for confirmation (unless auto-confirm is enabled)
    if auto_confirm:
        response = 'yes'
        print(f"Auto-confirming deletion of {len(orphaned)} files...")
    else:
        response = input(f"\nDelete all {len(orphaned)} orphaned files? (yes/no): ").strip().lower()
    
    if response == 'yes':
        deleted = 0
        failed = 0
        for filename in orphaned:
            filepath = os.path.join(AUDIO_FOLDER, filename)
            try:
                os.remove(filepath)
                deleted += 1
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")
                failed += 1
        
        print(f"\n✓ Cleanup complete!")
        print(f"  - Deleted: {deleted} files")
        if failed > 0:
            print(f"  - Failed: {failed} files")
        
        # Verify counts match now
        remaining_mp3s = len([f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith('.mp3')])
        print(f"\nFinal counts:")
        print(f"  - MP3 files: {remaining_mp3s}")
        print(f"  - CSV entries: {len(csv_filenames)}")
        
        if remaining_mp3s == len(csv_filenames):
            print(f"\n✓ SUCCESS! Counts now match perfectly!")
        else:
            print(f"\n⚠️ Warning: Counts still don't match (difference: {abs(remaining_mp3s - len(csv_filenames))})")
    else:
        print("\nCleanup cancelled. No files were deleted.")

if __name__ == "__main__":
    print("Orphaned MP3 File Cleanup Tool")
    print("=" * 60)
    
    # Check for --auto-confirm flag
    auto_confirm = '--auto-confirm' in sys.argv or '-y' in sys.argv
    
    cleanup_orphaned_files(auto_confirm=auto_confirm)
