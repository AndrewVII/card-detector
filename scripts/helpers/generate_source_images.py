import moviepy.editor as mp
import random
import os
from pytube import YouTube

YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=Jxi3AGQUTXc",
    "https://www.youtube.com/watch?v=r-_VLSiP6ww",
    "https://www.youtube.com/watch?v=Z7zfUkWKi-Q",
]
OUTPUT_PATH = "dataset"
OUTPUT_VIDEO_PATH = "videos"
OUTPUT_FRAMES_PATH = "source_images"
NUM_FRAMES = 20


def download_youtube_video(youtube_url, output_path, resolution="1080p"):
    yt = YouTube(youtube_url)
    # Filter streams by resolution
    stream = yt.streams.filter(res=resolution, file_extension="mp4").first()
    if not stream:
        print(
            f"No stream found for resolution {resolution}, downloading the highest resolution available."
        )
        stream = (
            yt.streams.filter(file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
        )
    video_path = stream.download(output_path=output_path)
    return video_path


def extract_random_frames(video_path, num_frames, output_dir, image_num_start):
    # Load the video
    video = mp.VideoFileClip(video_path)
    duration = video.duration  # Duration of the video in seconds

    # Generate random time stamps
    random_times = sorted(random.sample(range(int(duration)), num_frames))

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, t in enumerate(random_times):
        # Extract frame at the given time
        frame = video.get_frame(t)
        # Save the frame as an image
        frame_path = os.path.join(output_dir, f"image_{image_num_start+1}.jpg")
        mp.ImageClip(frame).save_frame(frame_path)
        print(f"Saved frame at {t} seconds as {frame_path}")
        image_num_start += 1


if __name__ == "__main__":
    frames_output_dir = os.path.join(OUTPUT_PATH, OUTPUT_FRAMES_PATH)
    video_output_dir = os.path.join(OUTPUT_PATH, OUTPUT_VIDEO_PATH)
    image_num = 0
    if os.path.exists(frames_output_dir):
        for file in os.listdir(frames_output_dir):
            image_num = max(image_num, int(file.split("_")[1].split(".")[0]))

    if os.path.exists(video_output_dir):
        for file in os.listdir(video_output_dir):
            os.remove(os.path.join(video_output_dir, file))

    for YOUTUBE_URL in YOUTUBE_URLS:
        video_path = download_youtube_video(YOUTUBE_URL, video_output_dir)
        extract_random_frames(video_path, NUM_FRAMES, frames_output_dir, image_num)
        image_num += NUM_FRAMES

    for file in os.listdir(video_output_dir):
        os.remove(os.path.join(video_output_dir, file))
    os.rmdir(video_output_dir)
