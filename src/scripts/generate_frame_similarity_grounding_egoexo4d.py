import pandas as pd
from transformers import ResNetModel
import torch
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import os
import argparse
from datetime import datetime
import cv2
import json
import ast
import pickle

def create_annotated_video(video_path, video_id, start_frame, stop_frame, fps, narration):
    if not os.path.exists(video_path):
        print(f"Error: Video path does not exist: {video_path}")
        return None
    output_filename = f"/private/home/arjunrs1/epic-sounds-annotations/visualization_dir/frame_similarity_grounding/{video_id}_{start_frame}_{stop_frame}.MP4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame-fps <= frame_counter <= stop_frame+fps:
            if start_frame <= frame_counter <= stop_frame:
                cv2.putText(frame, narration, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)
            writer.write(frame)
        frame_counter += 1
    
    cap.release()
    writer.release()
    return output_filename

def load_annotations(annotations_path, vid_id=None):
    with open(annotations_path, 'r') as file:
        data = json.load(file)
    if vid_id is not None:
        return data['annotations'][vid_id]
    return data['annotations']

def find_video_encoding(base_path, take_name):
    annotations_path = os.path.join(base_path, 'annotations', 'relations_val.json')
    annotations = load_annotations(annotations_path)
    for key, value in annotations.items():
        if value['take_name'] == take_name:
            return key
    return None

def convert_to_seconds(timestr):
    # Parse the time string to a datetime object
    time_obj = datetime.strptime(timestr, '%H:%M:%S.%f')
    # Calculate total seconds
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    return total_seconds

def extract_features_batch(video_path, model, batch_size=32, device='cuda'):
    # Ensure the model is on the correct device
    model = model.to(device)
    model.eval()
    # Define the image transformations
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = []
    frame_counter = 0
    all_features = []
    # Process the video frame by frame
    with tqdm(total=total_frames, desc="Extracting frames...") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if there are no frames left
            frame_list.append(preprocess(frame).unsqueeze(0))
            frame_counter += 1
            # Process in batches
            if len(frame_list) == batch_size or not ret:
                input_tensors = torch.cat(frame_list, dim=0).to(device)
                with torch.no_grad():
                    features = model(input_tensors).pooler_output.squeeze()
                    all_features.extend(features.cpu().detach().numpy())
                frame_list = []
                pbar.update(batch_size)
            if not ret:
                pbar.update(len(frame_list))
    cap.release()
    return frame_counter, all_features

def precompute_features(video_id, ego_video_path, exo_video_path, ego_cam_id, exo_cam_id, model, batch_size=32):
    features_dict = {}
    print(f"Processing {video_id}...")

    #generate ego features...
    num_frames, features = extract_features_batch(ego_video_path, model, batch_size=batch_size, device='cuda')
    for frame_index, feature in zip(range(num_frames), features):
        features_dict[(ego_cam_id, frame_index)] = feature

    #generate exo features...
    num_frames, features = extract_features_batch(exo_video_path, model, batch_size=batch_size, device='cuda')
    for frame_index, feature in zip(range(num_frames), features):
        features_dict[(exo_cam_id, frame_index)] = feature

    print(exo_cam_id)
    print(exo_video_path)
    print(ego_video_path)
    print(ego_cam_id)
    print(list(features_dict.keys())[:5])
    print(list(features_dict.keys())[-5:])
    print(len(list(features_dict.keys())))
    return features_dict
    
def find_interval_precomputed(frame_index, features_dict, camera_id, threshold):
    central_features = features_dict[(camera_id, frame_index)]
    
    left_boundary = frame_index
    right_boundary = frame_index
    
    # Check to the left
    while left_boundary > 1:
        left_features = features_dict.get((camera_id, left_boundary - 1))
        if left_features is None or cosine_similarity(torch.tensor(central_features), torch.tensor(left_features)) < threshold:
            break
        left_boundary -= 1
    
    # Check to the right
    while True:
        right_features = features_dict.get((camera_id, right_boundary + 1))
        if right_features is None or cosine_similarity(torch.tensor(central_features), torch.tensor(right_features)) < threshold:
            break
        right_boundary += 1
    
    return left_boundary, right_boundary

def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)

def main(args):
    tqdm.pandas()
    # Initialize the ResNet model
    model = ResNetModel.from_pretrained('microsoft/resnet-50')
    model.eval()

    base_path = '/datasets01/egoexo4d/v2'
    vid_id_encoding = find_video_encoding(base_path, args.take_id)
    base_vid_path = os.path.join(base_path, 'takes', args.take_id, 'frame_aligned_videos', 'downscaled', '448')
    ego_video_path = os.path.join(base_vid_path, args.ego_camera + '.mp4')
    exo_video_path = os.path.join(base_vid_path, args.exo_camera + '.mp4')

    # Check if features need to be recomputed or loaded from file
    feature_file = f'/private/home/arjunrs1/epic-sounds-annotations/per_frame_features/{args.take_id}_features.pkl'
    if os.path.exists(feature_file) and not args.recompute_features:
        print("Loading features from file...")
        with open(feature_file, 'rb') as f:
            features_dict = pickle.load(f)
    else:
        print("Computing features...")
        features_dict = precompute_features(args.take_id, ego_video_path, exo_video_path, args.ego_camera, args.exo_camera, model, batch_size=args.batch_size)
        with open(feature_file, 'wb') as f:
            pickle.dump(features_dict, f)

    # Load annotations and prepare data
    annotations_path = os.path.join(base_path, 'annotations', 'atomic_descriptions_val.json')
    narrations = load_annotations(annotations_path, vid_id_encoding)
    narrations = narrations[0]['descriptions']

    narrations_df = pd.DataFrame(narrations)
    # Select only the 'timestamp' and 'text' columns
    narrations_df = narrations_df[['timestamp', 'text']]

    # Apply the function to compute intervals
    narrations_df['narr_tmstp_frame'] = narrations_df['timestamp'].apply(lambda x: int(round(x * args.fps)))
    narrations_df['ego_interval'] = narrations_df.progress_apply(lambda row: find_interval_precomputed(row['narr_tmstp_frame'], features_dict, args.ego_camera, args.ego_view_threshold), axis=1)
    narrations_df['exo_interval'] = narrations_df.progress_apply(lambda row: find_interval_precomputed(row['narr_tmstp_frame'], features_dict, args.exo_camera, args.exo_view_threshold), axis=1)

    print(narrations_df.head())
    # Produce videos for sampled narrations
    if args.produce_ego_video or args.produce_exo_video:
        sampled_df = narrations_df.sample(n=args.num_videos, random_state=args.random_seed)
        for _, row in sampled_df.iterrows():
            try:
                ego_interval = ast.literal_eval(row['ego_interval'])
            except:
                ego_interval = row['ego_interval']
            try:
                exo_interval = ast.literal_eval(row['exo_interval'])
            except:
                exo_interval = row['exo_interval']

            if args.produce_ego_video:
                ego_output_filename = create_annotated_video(ego_video_path, f"{args.take_id}_{args.ego_camera}", ego_interval[0], ego_interval[1], args.fps, row['text'])
                print(f"Ego video produced: {ego_output_filename}")
            if args.produce_exo_video:
                exo_output_filename = create_annotated_video(exo_video_path, f"{args.take_id}_{args.exo_camera}", exo_interval[0], exo_interval[1], args.fps, row['text'])
                print(f"Exo video produced: {exo_output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute intervals based on frame similarity and IoU.')
    parser.add_argument('--ego_view_threshold', type=float, default=0.85, help='Similarity threshold for defining intervals on ego frames.')
    parser.add_argument('--exo_view_threshold', type=float, default=0.97, help='Similarity threshold for defining intervals on exo frames.')
    parser.add_argument('--fps', type=int, default=30, help='Video frame rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for precomputed feature learning.')
    parser.add_argument('--take_id', type=str, help='Take ID to filter videos (e.g. cmu_biking_01).')
    parser.add_argument('--recompute_features', action='store_true', help='Flag to recompute features if they dont exist.')
    parser.add_argument('--produce_ego_video', action='store_true', help='Whether to produce ego video of narration.')
    parser.add_argument('--produce_exo_video', action='store_true', help='Whether to produce exo video of narration.')
    parser.add_argument('--num_videos', type=int, default=10, help='Number of narration visualization videos to generate.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--ego_camera', type=str, default="aria04_214-1", help="Camera for ego view.")
    parser.add_argument('--exo_camera', type=str, default="cam01", help="Camera for exo view.")

    args = parser.parse_args()
    main(args)