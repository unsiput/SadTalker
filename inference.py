import torch
from time import strftime
import os, sys, time
from argparse import ArgumentParser
from pydub import AudioSegment
import subprocess
import ray  # Import Ray for parallel processing

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data

# Initialize Ray
ray.init()

def split_audio(audio_path, output_dir, segment_duration_ms=5000):
    """Splits audio into smaller segments."""
    from pydub import AudioSegment
    audio = AudioSegment.from_file(audio_path)
    os.makedirs(output_dir, exist_ok=True)
    segments = []
    for i, start_time in enumerate(range(0, len(audio), segment_duration_ms)):
        segment_path = os.path.join(output_dir, f"segment_{i}.wav")
        audio[start_time:start_time + segment_duration_ms].export(segment_path, format="wav")
        segments.append(segment_path)
    return segments

@ray.remote
def process_segment(segment_path, pic_path, args_template, segment_save_dir):
    """Processes a single audio segment using the original pipeline."""
    device = args_template["device"]
    preprocess_model = CropAndExtract(args_template["path_of_lm_croper"], 
                                      args_template["path_of_net_recon_model"], 
                                      args_template["dir_of_BFM_fitting"], device)
    audio_to_coeff = Audio2Coeff(args_template["audio2pose_checkpoint"], args_template["audio2pose_yaml_path"], 
                                 args_template["audio2exp_checkpoint"], args_template["audio2exp_yaml_path"], 
                                 args_template["wav2lip_checkpoint"], device)
    animate_from_coeff = AnimateFromCoeff(args_template["free_view_checkpoint"], 
                                          args_template["mapping_checkpoint"], 
                                          args_template["facerender_yaml_path"], device)

    # Create directories for each segment
    first_frame_dir = os.path.join(segment_save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)

    # Crop image and extract 3DMM from image
    first_coeff_path, crop_pic_path = preprocess_model.generate(pic_path, first_frame_dir)
    if first_coeff_path is None:
        raise RuntimeError("Can't get the coeffs of the input")

    # Generate coefficients from audio
    batch = get_data(first_coeff_path, segment_path, device)
    coeff_path = audio_to_coeff.generate(batch, segment_save_dir, args_template["pose_style"])

    # Generate video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, segment_path, 
                               args_template["batch_size"], 
                               args_template["camera_yaw"], 
                               args_template["camera_pitch"], 
                               args_template["camera_roll"],
                               expression_scale=args_template["expression_scale"], 
                               still_mode=args_template["still"])
    
    animate_from_coeff.generate(data, segment_save_dir, enhancer=args_template["enhancer"])
    return os.path.join(segment_save_dir, data["video_name"] + (("_enhanced" if args_template["enhancer"] else "") + ".mp4"))

def merge_videos(video_segments, output_path):
    """Merges video segments into a single video."""
    with open("file_list.txt", "w") as f:
        for segment in video_segments:
            f.write(f"file '{segment}'\n")
    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", "file_list.txt", "-c", "copy", output_path])
    os.remove("file_list.txt")

def main(args):
    # Prepare directories
    audio_segments_dir = "./audio_segments"
    video_segments_dir = "./video_segments"
    final_video = os.path.join(args.result_dir, "final_video.mp4")

    os.makedirs(video_segments_dir, exist_ok=True)

    # Split audio into segments
    audio_segments = split_audio(args.driven_audio, audio_segments_dir)

    # Define argument template for each segment
    args_template = {
        "device": args.device,
        "path_of_lm_croper": args.checkpoint_dir + '/shape_predictor_68_face_landmarks.dat',
        "path_of_net_recon_model": args.checkpoint_dir + '/epoch_20.pth',
        "dir_of_BFM_fitting": args.checkpoint_dir + '/BFM_Fitting',
        "audio2pose_checkpoint": args.checkpoint_dir + '/auido2pose_00140-model.pth',
        "audio2pose_yaml_path": './src/config/auido2pose.yaml',
        "audio2exp_checkpoint": args.checkpoint_dir + '/auido2exp_00300-model.pth',
        "audio2exp_yaml_path": './src/config/auido2exp.yaml',
        "wav2lip_checkpoint": args.checkpoint_dir + '/wav2lip.pth',
        "free_view_checkpoint": args.checkpoint_dir + '/facevid2vid_00189-model.pth.tar',
        "mapping_checkpoint": args.checkpoint_dir + '/mapping_00229-model.pth.tar',
        "facerender_yaml_path": './src/config/facerender.yaml',
        "pose_style": args.pose_style,
        "batch_size": args.batch_size,
        "camera_yaw": args.camera_yaw,
        "camera_pitch": args.camera_pitch,
        "camera_roll": args.camera_roll,
        "expression_scale": args.expression_scale,
        "still": args.still,
        "enhancer": args.enhancer
    }

    # Process each segment in parallel
    futures = [process_segment.remote(segment, args.source_image, args_template, video_segments_dir) for segment in audio_segments]
    video_segments = ray.get(futures)

    # Merge video segments
    merge_videos(video_segments, final_video)
    print(f"Final video saved at {final_video}")

if __name__ == '__main__':
    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./examples/driven_audio/japanese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/art_0.png', help="path to source image")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to checkpoints")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0, help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2, help="the batch size of facerender")
    parser.add_argument("--expression_scale", type=float, default=1.0, help="expression scale factor")
    parser.add_argument('--camera_yaw', nargs='+', type=int, default=[0], help="camera yaw degree")
    parser.add_argument('--camera_pitch', nargs='+', type=int, default=[0], help="camera pitch degree")
    parser.add_argument('--camera_roll', nargs='+', type=int, default=[0], help="camera roll degree")
    parser.add_argument('--enhancer', type=str, default=None, help="Face enhancer (e.g., GFPGAN)")
    parser.add_argument("--still", action="store_true", help="Still mode for animation")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="Use CPU instead of GPU")

    args = parser.parse_args()
    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    main(args)
