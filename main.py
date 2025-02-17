import time
import os
from yt_dlp import YoutubeDL

def download_videos(video_urls, ydl_opts):
    with YoutubeDL(ydl_opts) as ydl:
        for url in video_urls:
            success = False
            #Retry up to 3 times
            for attempt in range(3):
                try:
                    print(f"Attempting to download: {url} (Attempt {attempt+1})")
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

def main():
    #Create output directory (if it doesn't exist)
    os.makedirs('audio_files', exist_ok=True)

    #Read links from links.txt
    try:
        with open('links.txt', 'r', encoding='utf-8') as f:
            playlist_links = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("links.txt' not found")
        return

    for video_url in playlist_links:
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

        print(f"Processing URL/Playlist: {video_url}")
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                if 'entries' in info:  
                    #Playlist
                    playlist_title = info.get('title', 'Unknown Playlist')
                    print(f"Detected playlist: {playlist_title}")
                    entries = info['entries']

                    for i in range(0, len(entries), 10):
                        batch = entries[i:i + 10]
                        urls = [entry['webpage_url'] for entry in batch if entry]
                        download_videos(urls, ydl_opts)
                else:
                    #Single Video
                    title = info.get('title', 'Unknown Title')
                    print(f"Single video detected: {title}")
                    download_videos([info['webpage_url']], ydl_opts)

        except Exception as e:
            print(f"An error occurred while processing {video_url}: {e}")

if __name__ == '__main__':
    main()
