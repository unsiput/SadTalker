import os
import ray
from pydub import AudioSegment
import subprocess

ray.init()  # Initialize Ray

def split_audio(audio_path, output_dir, segment_duration_ms=5000):
    """Splits audio into smaller segments."""
    audio = AudioSegment.from_file(audio_path)
    os.makedirs(output_dir, exist_ok=True)
    segments = []
    for i, start_time in enumerate(range(0, len(audio), segment_duration_ms)):
        segment_path = os.path.join(output_dir, f"segment_{i}.wav")
        audio[start_time:start_time + segment_duration_ms].export(segment_path, format="wav")
        segments.append(segment_path)
    return segments

@ray.remote
def process_segment(segment_path, source_image, output_dir):
    """Processes a single audio segment using the model."""
    output_path = os.path.join(output_dir, os.path.basename(segment_path).replace(".wav", ".mp4"))
    os.makedirs(output_dir, exist_ok=True)
    os.system(f"python inference.py --driven_audio {segment_path} --source_image {source_image} --result_dir {output_dir}")
    return output_path

def merge_videos(video_segments, output_path):
    """Merges video segments into a single video."""
    with open("file_list.txt", "w") as f:
        for segment in video_segments:
            f.write(f"file '{segment}'\n")
    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", "file_list.txt", "-c", "copy", output_path])
    os.remove("file_list.txt")

def main(args):
    audio_segments_dir = "./audio_segments"
    video_segments_dir = "./video_segments"
    final_video = "./output/final_video.mp4"

    # Split audio into segments
    audio_segments = split_audio(args.driven_audio, audio_segments_dir)

    # Process segments in parallel using Ray
    os.makedirs(video_segments_dir, exist_ok=True)
    futures = [
        process_segment.remote(segment, args.source_image, video_segments_dir) 
        for segment in audio_segments
    ]
    video_segments = ray.get(futures)  # Gather results

    # Merge video segments
    merge_videos(video_segments, final_video)
    print(f"Final video saved at {final_video}")

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--driven_audio", default='./examples/driven_audio/japanese.wav', help="Path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/art_0.png', help="Path to source image")
    parser.add_argument("--result_dir", default='./results', help="Path to output directory")

    args = parser.parse_args()
    main(args)
