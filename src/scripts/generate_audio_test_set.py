import pandas as pd
import pickle
import os
import librosa
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation file with time intervals for sound event prediction.")
    parser.add_argument('--audio_dir', type=str, default="/private/home/arjunrs1/EPIC-SOUNDS", help='Directory where audio files are stored')
    parser.add_argument('--output_dir', type=str, default='/private/home/arjunrs1/epic-sounds-annotations/per-video-annotations', help='Output directory for annotations')
    parser.add_argument('--participant_id', type=str, help='Participant ID')
    parser.add_argument('--video_num', type=str, help='Video number')
    parser.add_argument('--sampling_rate', type=int, default=24000, help='Sampling rate for audio processing')
    parser.add_argument('--window_duration', type=float, default=1.0, help='Duration of each window in seconds')
    parser.add_argument('--hop_length', type=float, default=0.2, help='Hop length in seconds')

    args = parser.parse_args()

    video_id = f'{args.participant_id}_{args.video_num}'
    data, sr = librosa.load(os.path.join(args.audio_dir, video_id + '.wav'), sr=args.sampling_rate)
    total_duration = int(len(data) / sr)

    # Calculate the number of annotations to create
    num_annotations = int((total_duration - args.window_duration) / args.hop_length) + 1

    # Generate data
    new_data = []
    print("Generating time intervals...")
    for i in tqdm(range(num_annotations)):
        start_time = i * args.hop_length
        stop_time = start_time + args.window_duration
        annotation_id = f'{video_id}_{i}'

        start_sample = int(start_time * args.sampling_rate)
        stop_sample = int(stop_time * args.sampling_rate)

        start_timestamp_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{start_time % 60:06.3f}"
        stop_timestamp_str = f"{int(stop_time // 3600):02}:{int((stop_time % 3600) // 60):02}:{stop_time % 60:06.3f}"

        new_data.append({
            'annotation_id': annotation_id,
            'participant_id': args.participant_id,
            'video_id': video_id,
            'start_timestamp': start_timestamp_str,
            'stop_timestamp': stop_timestamp_str,
            'start_sample': start_sample,
            'stop_sample': stop_sample
        })

    # Create DataFrame
    new_annotations_df = pd.DataFrame(new_data)

    # Specify the file path
    tim_formatted = args.window_duration == 1.0
    if not os.path.exists(os.path.join(args.output_dir, video_id)):
        os.makedirs(os.path.join(args.output_dir, video_id))
    new_annotations_file = f'{args.output_dir}/{video_id}/EPIC_Sounds_recognition_{video_id}_win={args.window_duration}_hop={args.hop_length}_test_timestamps.pkl'
    if tim_formatted:
        new_annotations_file = f'{args.output_dir}/{video_id}/EPIC_Sounds_recognition_{video_id}_win={args.window_duration}_hop={args.hop_length}_test_timestamps_tim_formatted.pkl'

    # Save the DataFrame to the file path
    new_annotations_df.to_pickle(new_annotations_file)
    print("Done")

if __name__ == "__main__":
    main()