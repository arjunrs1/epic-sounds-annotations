import torch
import pandas as pd
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.special import softmax
import os
import argparse
import librosa

# Convert timestamps to seconds for plotting
def timestamp_to_seconds(timestamp):
    return timestamp.dt.hour * 3600 + timestamp.dt.minute * 60 + timestamp.dt.second + timestamp.dt.microsecond / 1e6

def generate_frame_based_grounding(narrations_df, similarity_scores, feature_duration, feature_stride, similarity_threshold=0.3):
    narrations = []
    start_times = []
    end_times = []
    for idx, row in narrations_df.iterrows():
        start = row['narration_seconds']
        end = row['narration_seconds']
        
        while True:
            expand_left = expand_right = False
            # Calculate indices based on the stride
            left_index = int((start - feature_stride) / feature_stride)
            right_index = int((end + feature_stride) / feature_stride)
            
            # Check if expansion is possible to the left
            if left_index >= 0 and similarity_scores[idx][left_index] >= similarity_threshold:
                start -= feature_duration  # Move start back by the duration of one feature
                expand_left = True
            
            # Check if expansion is possible to the right
            if right_index < similarity_scores.shape[1] and similarity_scores[idx][right_index] >= similarity_threshold:
                end += feature_duration  # Move end forward by the duration of one feature
                expand_right = True
            
            # Break the loop if neither side can be expanded
            if not expand_left and not expand_right:
                break
        
        narrations.append(row['narration_id'])
        start_times.append(start)
        end_times.append(end)
        """ print(f"Expanded Ground Truth Interval: {start:.2f} to {end:.2f} seconds")
        print(f"Ground Truth Narration: {row['narration']} at {row['narration_timestamp']}")
        print() """
    return narrations, start_times, end_times

def generate_similarity_scores(narration_features, video_features, use_qual_vis=False):
    if use_qual_vis:
        return cosine_similarity(narration_features, video_features, dim=1)
    else:
        dot_product = torch.matmul(narration_features, video_features.transpose(-1, -2))
        narration_norm = torch.linalg.norm(narration_features, dim=-1, keepdim=True)
        video_norm = torch.linalg.norm(video_features, dim=-1, keepdim=True)
        return torch.div(dot_product, torch.mul(narration_norm, video_norm.T))
    
def generate_qual_vis(narrations_df, similarity_scores, feature_duration, feature_stride, narr_idx, top_n=5):
    top_indices = torch.topk(similarity_scores, top_n).indices.flatten()
    print(top_indices)
    start_times = top_indices * feature_stride  # Use feature_stride for calculating start times
    end_times = start_times + feature_duration  # End times are start times plus the duration
    print("Current narration:", narrations_df.iloc[narr_idx]['narration'])
    print(f"Ground Truth Interval: {narrations_df.iloc[narr_idx]['start_seconds']:.2f} to {narrations_df.iloc[narr_idx]['stop_seconds']:.2f} seconds")
    for start, end in zip(start_times.numpy().flatten(), end_times.numpy().flatten()):
        overlapping_narrations = narrations_df[(narrations_df['start_seconds'] <= end) & (narrations_df['stop_seconds'] >= start)]
        print(len(overlapping_narrations))
        print(f"High Similarity Interval: {start:.2f} to {end:.2f} seconds")
        if not overlapping_narrations.empty:
            print("Overlapping Ground Truth Narrations:")
            for _, row in overlapping_narrations.iterrows():
                print(f"  - {row['narration']} from {row['start_timestamp']} to {row['stop_timestamp']}")
        else:
            print("  - No overlapping narrations.")
        print()

def main():
    parser = argparse.ArgumentParser(description="Process video and narration features to generate grounding based on similarity scores.")
    parser.add_argument('--participant_id', type=str, help='Participant ID')
    parser.add_argument('--video_num', type=str, help='Video number')
    parser.add_argument('--annotations_file', type=str, default='/private/home/arjunrs1/epic-kitchens-100-annotations/EPIC_100_train.csv', help='CSV file containing annotations')
    parser.add_argument('--output_dir', type=str, default="/private/home/arjunrs1/epic-sounds-annotations/video_grounded_narrations", help='Output directory for grounded narrations')
    parser.add_argument('--similarity_threshold', type=float, default=0.3, help='Threshold for similarity to consider grounding')
    parser.add_argument('--feature_duration', type=float, default=1.0, help='Duration of each video feature')
    parser.add_argument('--feature_stride', type=float, default=0.5, help='Stride between consecutive video features')
    parser.add_argument('--qual_vis', action='store_true', help='Flag to enable qualitative visualization')
    parser.add_argument('--top_n', type=int, default=5, help='Number of top similar segments to display (for qual_vis only)')
    parser.add_argument('--narr_index', type=int, default=0, help='Index of the narration for qualitative visualization')
    parser.add_argument('--N', type=int, default=None, help='Number of seconds to trim the audio')
    parser.add_argument('--sr', type=int, default=24000, help='Audio sampling rate')
    args = parser.parse_args()

    video_id = f"{args.participant_id}_{args.video_num}"

    #trim if necessary
    if args.N is None:
        data, sr = librosa.load(os.path.join("/private/home/arjunrs1/EPIC-SOUNDS", video_id + '.wav'), sr=args.sr)
        args.N = int(len(data) / sr)

    # Load narrations
    narrations_df = pd.read_csv(args.annotations_file)
    narrations_df = narrations_df[narrations_df.video_id == video_id]
    narrations_df = narrations_df.sort_values(by='start_timestamp')
    narrations_df['start_seconds'] = timestamp_to_seconds(pd.to_datetime(narrations_df['start_timestamp']))
    narrations_df['stop_seconds'] = timestamp_to_seconds(pd.to_datetime(narrations_df['stop_timestamp']))
    narrations_df['narration_timestamp'] = pd.to_datetime(narrations_df['narration_timestamp'])
    narrations_df['narration_seconds'] = timestamp_to_seconds(narrations_df['narration_timestamp'])
    narrations_df = narrations_df[narrations_df.stop_seconds <= args.N]
    narrations_df.reset_index(inplace=True, drop=True)

    # Load video features
    video_features_dir = f"/private/home/arjunrs1/EgoVLPv2/EgoVLPv2/video_features/{video_id}"
    video_features = []
    for feature_file_name in os.listdir(video_features_dir):
        video_features.append(torch.load(os.path.join(video_features_dir, feature_file_name)))
    video_features = torch.concat(video_features)

    #load narration_features
    if args.qual_vis:
        first_narration_id = narrations_df.iloc[args.narr_index]['narration_id']
        narration_features = torch.load(f'/private/home/arjunrs1/EgoVLPv2/EgoVLPv2/narration_features/{video_id}/{first_narration_id}.pt').cpu()
    else:
        narration_features = []
        for _, row in narrations_df.iterrows():
            narration_features.append(torch.load(f'/private/home/arjunrs1/EgoVLPv2/EgoVLPv2/narration_features/{video_id}/{row.narration_id}.pt').cpu())
        narration_features = torch.concat(narration_features)
    narration_features = narration_features.unsqueeze(0) if narration_features.dim() == 1 else narration_features

    similarity_scores = generate_similarity_scores(narration_features, video_features, use_qual_vis=args.qual_vis)
    if args.qual_vis:
        generate_qual_vis(narrations_df, similarity_scores, args.feature_duration, args.feature_stride, args.narr_index, top_n=args.top_n)
    else:
        narrations, start_times, end_times = generate_frame_based_grounding(narrations_df, similarity_scores, args.feature_duration, args.feature_stride, similarity_threshold=args.similarity_threshold)

        narration_grounded = pd.DataFrame({
            'narration_id': narrations,
            'start_seconds': start_times,
            'stop_seconds': end_times
        })
        
        #Post-processing to format as the audio_grounded narrations df is:
        narration_grounded['assigned_intervals'] = narration_grounded.apply(lambda row: [[row['start_seconds'], row['stop_seconds']]], axis=1)
        narration_grounded = narration_grounded.drop(['start_seconds', 'stop_seconds'], axis=1)
        merged_df = pd.merge(narrations_df, narration_grounded, on='narration_id')
        merged_df = merged_df[['start_timestamp', 'stop_timestamp', 'narration', 'start_seconds', 'stop_seconds', 'assigned_intervals']]
        merged_df = merged_df.sort_values(by='start_timestamp')
        merged_df.reset_index(inplace=True, drop=True)

        #save narration grouding to file:
        if not os.path.exists(os.path.join(args.output_dir, video_id)):
            os.makedirs(os.path.join(args.output_dir, video_id))
        video_grounded_narrs_filename = f"similarity_threshold={args.similarity_threshold}_feature_duration={args.feature_duration}_feature_stride={args.feature_stride}.pkl"
        video_grounded_narrations_filepath = os.path.join("/private/home/arjunrs1/epic-sounds-annotations", args.output_dir, video_id, video_grounded_narrs_filename)
        merged_df.to_pickle(video_grounded_narrations_filepath)

if __name__ == "__main__":
    main()


#TODO: Modify video feature generation:
    #3) Use EK-100 finetuned checkpoint instead of pre-trained checkpoint.
#TODO: Show that with more fine-grained features, we immprove grounding (going from coarse 2 second features to more-finegrained 1 second featuures with 0.5 overlap).