import pandas as pd
from PIL import Image
from transformers import ResNetModel, ResNetConfig
import torch
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import os
import argparse
import glob
from datetime import datetime
import pickle
import cv2

def create_annotated_video(video_path, video_id, start_frame, stop_frame, fps, narration):
    output_filename = f"/private/home/arjunrs1/epic-sounds-annotations/visualization_dir/frame_similarity_grounding/{video_id}_{start_frame}_{stop_frame}.MP4"
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get the video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a writer
    writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Iterate through the frames
    frame_counter = 0
    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the frame is within the predicted narration interval
        if start_frame-fps <= frame_counter <= stop_frame+fps:
            # Add the text overlay
            if start_frame <= frame_counter <= stop_frame:
                cv2.putText(frame, narration, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)

            # Write the frame to the output video
            writer.write(frame)

        # Increment the frame counter
        frame_counter += 1

    # Release resources
    cap.release()
    writer.release()

    return output_filename

def convert_to_seconds(timestr):
    # Parse the time string to a datetime object
    time_obj = datetime.strptime(timestr, '%H:%M:%S.%f')
    # Calculate total seconds
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    return total_seconds

def extract_features_batch(image_paths, model, batch_size=32, device='cuda'):
    # Ensure the model is on the correct device
    model = model.to(device)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Initialize an empty list to store features
    all_features = []
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(img_path).convert('RGB') for img_path in batch_paths]
        input_tensors = torch.stack([preprocess(img) for img in images]).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensors).pooler_output.squeeze()
            all_features.extend(outputs.cpu().detach().numpy())  # Move tensors to CPU after processing
    
    return all_features

def precompute_features(video_ids, base_path, model, batch_size=32):
    features_dict = {}
    for video_id in video_ids:
        print(f"Processing {video_id}...")
        frame_paths = sorted(glob.glob(os.path.join(base_path, video_id.split("_")[0], "rgb_frames", video_id, '*.jpg')))
        features = extract_features_batch(frame_paths, model, batch_size=batch_size, device='cuda')
        for frame_path, feature in zip(frame_paths, features):
            frame_index = int(os.path.basename(frame_path).split('_')[-1].split('.')[0])
            features_dict[(video_id, frame_index)] = feature
    return features_dict

def find_interval_precomputed(frame_index, video_id, features_dict, threshold):
    central_features = features_dict[(video_id, frame_index)]
    
    left_boundary = frame_index
    right_boundary = frame_index
    
    # Check to the left
    while left_boundary > 1:
        left_features = features_dict.get((video_id, left_boundary - 1))
        if left_features is None or cosine_similarity(torch.tensor(central_features), torch.tensor(left_features)) < threshold:
            break
        left_boundary -= 1
    
    # Check to the right
    while True:
        right_features = features_dict.get((video_id, right_boundary + 1))
        if right_features is None or cosine_similarity(torch.tensor(central_features), torch.tensor(right_features)) < threshold:
            break
        right_boundary += 1
    
    return left_boundary, right_boundary

def iou(interval1, interval2):
    start_max = max(interval1[0], interval2[0])
    end_min = min(interval1[1], interval2[1])
    intersection = max(0, end_min - start_max)
    union = max(interval1[1], interval2[1]) - min(interval1[0], interval2[0])
    return intersection / union if union != 0 else 0

def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)

def main(args):
    tqdm.pandas()
    # Load the CSV file
    df = pd.read_csv('/private/home/arjunrs1/epic-kitchens-100-annotations/EPIC_100_train.csv')
    # Initialize the ResNet model
    model = ResNetModel.from_pretrained('microsoft/resnet-50')
    model.eval()

    base_path = '/datasets01/EPIC-KITCHENS-100'
    video_ids = [args.video_id]  # Example video IDs
    feature_file = f'/private/home/arjunrs1/epic-sounds-annotations/per_frame_features/{args.video_id}.pkl'

    if os.path.exists(feature_file) and not args.recompute_features:
        print("Loading features from file...")
        with open(feature_file, 'rb') as f:
            features_dict = pickle.load(f)
    else:
        print("Computing features...")
        features_dict = precompute_features(video_ids, base_path, model, batch_size=args.batch_size)
        with open(feature_file, 'wb') as f:
            pickle.dump(features_dict, f)

    # Apply the function to compute intervals
    df = df[df.video_id.isin(video_ids)]
    df['narr_tmstp_seconds'] = df['narration_timestamp'].apply(convert_to_seconds)
    df['narr_tmstp_frame'] = df['narr_tmstp_seconds'].apply(lambda x: int(round(x * args.fps)))
    df['interval'] = df.progress_apply(lambda row: find_interval_precomputed(row['narr_tmstp_frame'], row['video_id'], features_dict, args.threshold), axis=1)
    # Compute IoU
    df['iou'] = df.progress_apply(lambda row: iou(row['interval'], (row['start_frame'], row['stop_frame'])), axis=1)

    # Assuming 'df' is your DataFrame and it has a column 'iou' with IoU values
    thresholds = [0.1, 0.3, 0.5, 0.7]
    results = {}

    for threshold in thresholds:
        # Calculate the number of entries with IoU greater than the current threshold
        count_above_threshold = (df['iou'] > threshold).sum()
        # Calculate the percentage
        percentage_above_threshold = (count_above_threshold / len(df)) * 100
        results[threshold] = percentage_above_threshold

    # Print the results
    for threshold, percentage in results.items():
        print(f"Percentage of narrations with IoU > {threshold}: {percentage:.2f}%")
    print(f"Mean narration IoU: {df['iou'].mean()}")

    # Produce videos for sampled narrations
    if args.produce_video:
         # Produce the video if the flag is set
        video_path = f"/datasets01/EPIC-KITCHENS-100-VIDEOS-ht256px/060122/{args.video_id.split('_')[0]}/{args.video_id}.MP4"
        # Randomly sample 10 narrations to produce videos
        sampled_df = df.sample(n=args.num_videos, random_state=args.random_seed)
        for _, row in sampled_df.iterrows():
            output_filename = create_annotated_video(video_path, args.video_id, row['interval'][0], row['interval'][1], args.fps, row['narration'])
            print(f"Video produced: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute intervals based on frame similarity and IoU.')
    parser.add_argument('--threshold', type=float, default=0.65, help='Similarity threshold for defining intervals.')
    parser.add_argument('--fps', type=int, default=59.94, help='Video frame rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for precomputed feature learning.')
    parser.add_argument('--video_id', type=str, help='Video id.')
    parser.add_argument('--recompute_features', action='store_true', help='Flag to recompute features if they dont exist.')
    parser.add_argument('--produce_video', action='store_true', help='Whether to produce video of narration.')
    parser.add_argument('--num_videos', type=int, default=10, help='Number of narration visualization videos to generate.')
    parser.add_argument('--random_seed', type=int, default=41, help='Random seed for reproducibility.')   
    args = parser.parse_args()
    main(args)