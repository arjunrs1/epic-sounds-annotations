import numpy as np
import os
import random
import pickle
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
import librosa
import librosa.display
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.stats import mode
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import torch.nn.functional as F

def cluster_and_predict(class_confidences, window_size, hop_length, dendrogram_height, sampling_rate=24000):
    # Compute the Euclidean distance matrix between all pairs of samples
    dist_matrix = np.sqrt(np.sum((class_confidences[:, None, :] - class_confidences) ** 2, axis=-1))
    
    # Perform agglomerative clustering using complete linkage and Euclidean distance
    Z = linkage(dist_matrix, method='complete')
    
    # Cut the dendrogram at a height of 0.5 to obtain contiguous clusters
    clusters = cut_tree(Z, height=dendrogram_height)
    
    # Compute the mode max-confidence class within each cluster
    cluster_labels = clusters.reshape(-1)
    cluster_counts = np.bincount(cluster_labels)
    cluster_max_confidences = np.empty(len(cluster_counts), dtype=int)
    for i in range(len(cluster_counts)):
        cluster_mask = cluster_labels == i
        cluster_confidences = class_confidences[cluster_mask]
        cluster_max_confidences[i] = mode(np.argmax(cluster_confidences, axis=1)).mode
        
    # Compute the start and end times for each cluster
    cluster_starts = {}
    cluster_ends = {}
    for i in range(len(cluster_labels)):
        cluster_id = cluster_labels[i]
        if cluster_id not in cluster_starts:
            cluster_starts[cluster_id] = i
            cluster_ends[cluster_id] = i
        elif i > cluster_ends[cluster_id]:
            cluster_ends[cluster_id] = i
            
    # Convert the indices to time values
    start_times = np.array([cluster_starts[c]*hop_length for c in cluster_starts])
    end_times = np.array([cluster_ends[c]*hop_length+window_size for c in cluster_ends])
    
    # Round the start and end times to the nearest multiple of the hop length
    round_precision = hop_length * np.sign(hop_length)
    start_times = np.round(start_times / round_precision) * round_precision
    end_times = np.round(end_times / round_precision) * round_precision
    
    return cluster_max_confidences, start_times, end_times


def hard_merge_and_predict(predictions, probabilities, window_size, hop_length, audio_event_detection_threshold=0.5, sampling_rate=24000):

    unique_predictions = []
    start_times = []
    end_times = []
    average_confidences = []

    current_pred = predictions[0]
    start_time = 0
    segment_indices = [0]  # Start with the first index

    for i, pred in enumerate(predictions[1:], start=1):
        if pred != current_pred:
            # Calculate average confidence for the current segment
            segment_confidences = probabilities[segment_indices, current_pred]
            average_confidence = np.mean(segment_confidences)

            # Append results for this segment *IF IT MEETS DETECTION CONFIDENCE THRESHOLD*:
            if average_confidence >= audio_event_detection_threshold:
                unique_predictions.append(current_pred)
                start_times.append(start_time)
                end_times.append((i - 1) * hop_length + window_size)
                average_confidences.append(average_confidence)

            # Reset for the next segment
            current_pred = pred
            start_time = i * hop_length
            segment_indices = []

        segment_indices.append(i)

    # Handle the last segment
    segment_confidences = probabilities[segment_indices, current_pred]
    average_confidence = np.mean(segment_confidences)
    if average_confidence >= audio_event_detection_threshold:
        unique_predictions.append(current_pred)
        start_times.append(start_time)
        end_times.append((len(predictions) - 1) * hop_length + window_size)
        average_confidences.append(average_confidence)

    return unique_predictions, start_times, end_times, average_confidences

def llama_audio_narration_assignment(narrations, audio_event, tokenizer, llama3_model):
    examples = [
        ("cut zucchini", "cut/chop"),
        ("wash carrot", "water"),
        ("open bag", "rustle"),
    ]
    prompt = f"Given the following examples of actions and sounds they will produce:\n\n"
    for narration, sound in examples:
        prompt += f"* {narration} -> {sound}\n"
    prompt += "\nIdentify which of the following actions is most likely to produce the given sound, or indicate if None of them will.\n\n"
    audio_event = audio_event
    task_prompt = f"{prompt} the action(s) are [{', '.join(narrations)}], and the sound is {audio_event}."
    input_ids = tokenizer(task_prompt, return_tensors="pt").input_ids.to(device="cuda")
    output_ids = llama3_model.generate(input_ids, max_new_tokens=4, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    parsed_answer = output.split(": ")[-1].split("->")[-1].strip()
    print("INPUT:")
    print(task_prompt)
    if "none" in parsed_answer.lower():
        parsed_answer = None
    print("OUTPUT:")
    print(parsed_answer)
    return output, parsed_answer

def find_closest_narrations(audio_event_row, narrations_df):
    before_narrations = narrations_df[narrations_df['stop_seconds'] <= audio_event_row['start_seconds']]
    after_narrations = narrations_df[narrations_df['start_seconds'] >= audio_event_row['stop_seconds']]

    closest_before = before_narrations.iloc[-1] if not before_narrations.empty else None
    closest_after = after_narrations.iloc[0] if not after_narrations.empty else None
    return closest_before, closest_after

def assign_audio_events(narrations_df, audio_events_df, narr_embeds, audio_embeds, text_similarity_model, mode='best', use_llama_assignment=False, audio_narr_sim_thresh=0.4, tokenizer=None, llama3_model=None):
    assignment = {}
    for audio_idx, audio_event_row in tqdm(audio_events_df.iterrows(), total=audio_events_df.shape[0]):
        #find narrations that overlap with the audio event
        overlaps = []
        if use_llama_assignment:
            llama_narrations = []
        for nar_idx, nar_row in narrations_df.iterrows():
            if nar_row['start_seconds'] <= audio_event_row['stop_seconds'] and nar_row['stop_seconds'] >= audio_event_row['start_seconds']:
                if use_llama_assignment:
                    llama_narrations.append((nar_row.narration, nar_idx))
                else:
                    similarity = cosine_similarity(narr_embeds[nar_idx].reshape(1, -1).cpu(), audio_embeds[audio_idx].reshape(1, -1).cpu())
                    overlaps.append((nar_idx, similarity))
        if use_llama_assignment and llama_narrations:
            full_output, best_narration = llama_audio_narration_assignment([narr[0] for narr in llama_narrations], audio_event_row.description, tokenizer=tokenizer, llama3_model=llama3_model)
            if best_narration is not None:
                #embed the best narration, compare to each of the narration embeddings, and store overlaps as (nar_idx, similarity)
                llama_generated_narr = text_similarity_model.encode([best_narration], convert_to_tensor=True)#.cpu()
                for narr, nar_idx in llama_narrations:
                    #similarity = cosine_similarity(narr_embeds[nar_idx].reshape(1, -1), llama_generated_narr.reshape(1, -1))[0][0]
                    similarity = F.cosine_similarity(narr_embeds[nar_idx].unsqueeze(0), llama_generated_narr, dim=1).item()                    
                    overlaps.append((nar_idx, similarity))
                
        if overlaps:
            if mode=='best':
                best_narration = max(overlaps, key=lambda x: x[1])
                if best_narration[1] < audio_narr_sim_thresh:
                    best_narration = (None, None)
            elif mode=='random':
                best_narration = random.choice(overlaps)
                if best_narration[1] < audio_narr_sim_thresh:
                        best_narration = (None, None)
            assignment[audio_idx] = best_narration
        else:
            closest_before, closest_after = find_closest_narrations(audio_event_row, narrations_df)
            closest_narrations = []
            if use_llama_assignment:
                llama_closest = []
            if closest_before is not None:
                if use_llama_assignment:
                    llama_closest.append((closest_before.narration, closest_before.name))
                else:
                    sim_before = cosine_similarity(narr_embeds[closest_before.name].reshape(1, -1).cpu(), 
                                                audio_embeds[audio_idx].reshape(1, -1).cpu())[0][0]
                    closest_narrations.append((closest_before.name, sim_before))
            if closest_after is not None:
                if use_llama_assignment:
                    llama_closest.append((closest_after.narration, closest_after.name))
                else:
                    sim_after = cosine_similarity(narr_embeds[closest_after.name].reshape(1, -1).cpu(), 
                                                audio_embeds[audio_idx].reshape(1, -1).cpu())[0][0]
                    closest_narrations.append((closest_after.name, sim_after))

            if use_llama_assignment and llama_closest:
                full_output, best_narration = llama_audio_narration_assignment([narr[0] for narr in llama_closest], audio_event_row.description, tokenizer=tokenizer, llama3_model=llama3_model)
                if best_narration is not None:
                    llama_generated_narr = text_similarity_model.encode([best_narration], convert_to_tensor=True)#.cpu()
                    for narr, nar_idx in llama_closest:
                        #similarity = cosine_similarity(narr_embeds[nar_idx].reshape(1, -1), llama_generated_narr.reshape(1, -1))[0][0]
                        similarity = F.cosine_similarity(narr_embeds[nar_idx].unsqueeze(0), llama_generated_narr, dim=1).item()
                        print(narr_embeds[nar_idx].shape)
                        print(llama_generated_narr.shape)
                        print(similarity)
                        closest_narrations.append((nar_idx, similarity))

            # If no narrations before or after, choose the closest available
            if not closest_narrations:
                if closest_before is not None:
                    assignment[audio_idx] = (closest_before.name, None)
                elif closest_after is not None:
                    assignment[audio_idx] = (closest_after.name, None)
            else:
                if mode=='best':
                    best_narration = max(closest_narrations, key=lambda x: x[1])
                    if best_narration[1] < audio_narr_sim_thresh:
                        best_narration = (None, None)
                elif mode=='random':
                    best_narration = random.choice(closest_narrations)
                    if best_narration[1] < audio_narr_sim_thresh:
                        best_narration = (None, None)
                assignment[audio_idx] = best_narration

    return assignment

def main():
    parser = argparse.ArgumentParser(description="Process audio annotations and generate outputs based on various parameters.")
    
    # File-loading options
    parser.add_argument('--output_dir', type=str, default="/private/home/arjunrs1/epic-sounds-annotations/src/outputs", help='Output directory for processed files')
    parser.add_argument('--audio_annotations_train_file', type=str, default="/private/home/arjunrs1/epic-sounds-annotations/EPIC_Sounds_train.pkl", help='Path to the training audio annotations file')
    parser.add_argument('--audio_annotations_val_file', type=str, default="/private/home/arjunrs1/epic-sounds-annotations/EPIC_Sounds_validation.pkl", help='Path to the validation audio annotations file')
    parser.add_argument('--class_mapping_file', type=str, default='/private/home/arjunrs1/epic-sounds-annotations/annotation_mapping.pkl', help='Path to the class mapping file')
    parser.add_argument('--participant_id', type=str, help='Participant ID')
    parser.add_argument('--video_num', type=str, help='Video number')
    
    # Other options
    parser.add_argument('--win_length', type=float, default=1.0, help='Window length for audio feature extraction')
    parser.add_argument('--hop_length', type=float, default=0.2, help='Hop length for audio feature extraction')
    parser.add_argument('--use_clustered_bounds', action='store_true', help='Flag to use clustered bounds for computing audio events')
    parser.add_argument('--audio_narration_similarity_threshold', type=float, default=0.4, help='Similarity threshold for assigning audio to narrations')
    parser.add_argument('--dendrogram_height', type=float, default=0.8, help='Height of dendrogram to cut for clustering')
    parser.add_argument('--use_llama_assignment', action='store_true', help='Flag to use llama-based assignment of audio to narrations')
    parser.add_argument('--N', type=int, default=None, help='Number of seconds to trim the audio')
    parser.add_argument('--multi_interval_assignment', action='store_true', help='Flag to determine how to merge assigned intervals')
    parser.add_argument('--audio_assignment_scheme', type=str, default="best", choices=['best', 'random'], help='Scheme for assigning audio to narrations')
    parser.add_argument('--audio_detection_threshold', type=float, default=0.0, help='Threshold confidence for audio event to be detected')
    
    args = parser.parse_args()

    # Construct video_id and output_file using provided arguments:
    video_id = f"{args.participant_id}_{args.video_num}"
    tim_formatted = "_tim_formatted" if args.win_length == 1.0 else ""
    output_file = f"scores/EPIC_Sounds_recognition_{video_id}_win={args.win_length}_hop={args.hop_length}_test_timestamps{tim_formatted}.pkl"
    
    #load in sBERT model:
    text_similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda")

    #load in llama model if necessary:
    if args.use_llama_assignment:
        hf_token = "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"
        #tokenizer = AutoTokenizer.from_pretrained("/large_experiments/ram/shared/Meta-Llama-3-8B-Instruct-hf")
        #llama3_model = AutoModelForCausalLM.from_pretrained("/large_experiments/ram/shared/Meta-Llama-3-8B-Instruct-hf", device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", use_auth_token=hf_token, low_cpu_mem_usage=True)
        llama3_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", device_map="cuda", use_auth_token=hf_token, low_cpu_mem_usage=True)

    #trim if necessary
    if args.N is None:
        data, sr = librosa.load(os.path.join("/private/home/arjunrs1/EPIC-SOUNDS", video_id + '.wav'), sr=24000)
        args.N = int(len(data) / sr)

    #load in predictions and labels
    with open(os.path.join(args.output_dir, f"{video_id}_win={args.win_length}_hop={args.hop_length}", output_file), 'rb') as f:
        pred_annotations = pickle.load(f)
    probabilities = softmax(pred_annotations['interaction_output'], axis=1)
    pred_labels = np.argsort(-probabilities, axis=1)[:, :1]

    # Load audio file
    audio_path = f'/private/home/arjunrs1/EPIC-SOUNDS/{args.participant_id}_{args.video_num}.wav'
    audio, sr = librosa.load(audio_path, sr=None)
    N_seconds_samples = args.N * sr
    audio = audio[:N_seconds_samples]

    # Load class predictions
    predictions=pred_labels.flatten()[:int((args.N-args.win_length)/args.hop_length)+1]  # Load and trim
    # Load and truncate class probabilities
    probabilities=probabilities[:int((args.N-args.win_length)/args.hop_length)+1,:]  # Load and trim
    # Load class mapping
    with open(args.class_mapping_file, 'rb') as file:
        class_mapping = pickle.load(file)

    #load in audio ground truth annotations:
    audio_annotations_train = pd.read_pickle(args.audio_annotations_train_file)
    audio_annotations_val = pd.read_pickle(args.audio_annotations_val_file)
    audio_annotations = pd.concat([audio_annotations_train, audio_annotations_val], axis=0)

    #Process annotations for target video:
    audio_annotations_gt = audio_annotations[audio_annotations.video_id==video_id]
    audio_annotations_gt = audio_annotations_gt[['start_timestamp', 'stop_timestamp', 'description', 'class', 'class_id']].copy()
    audio_annotations_gt['start_seconds'] = pd.to_datetime(audio_annotations_gt['start_timestamp'], format='%H:%M:%S.%f').dt.time
    audio_annotations_gt['stop_seconds'] = pd.to_datetime(audio_annotations_gt['stop_timestamp'], format='%H:%M:%S.%f').dt.time
    audio_annotations_gt['start_seconds'] = audio_annotations_gt['start_seconds'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)
    audio_annotations_gt['stop_seconds'] = audio_annotations_gt['stop_seconds'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)

    #Compute audio events and their temporal bounds
    if args.use_clustered_bounds:
        unique_predictions, start_times, end_times = cluster_and_predict(probabilities, window_size=args.win_length, 
                                                                         hop_length=args.hop_length, dendrogram_height=args.dendrogram_height)
    else:
        # Identify contiguous audio segments and calculate their intervals (get audio events and their grounding intervals)
        unique_predictions, start_times, end_times, event_confs = hard_merge_and_predict(predictions, probabilities, window_size=args.win_length, 
                                                                               hop_length=args.hop_length, audio_event_detection_threshold=args.audio_detection_threshold)

    print(f"Number of audio events: {len(unique_predictions)}")
    #Load in narrations and ground truth bounds
    narrations_df = pd.read_csv('~/epic-kitchens-100-annotations/EPIC_100_train.csv')
    narrations_df = narrations_df[(narrations_df['participant_id'] == args.participant_id) & (narrations_df['video_id'] == video_id)]
    narrations = narrations_df[['start_timestamp', 'stop_timestamp', 'narration', 'narration_timestamp']].copy()
    narrations['start_seconds'] = pd.to_datetime(narrations['start_timestamp'], format='%H:%M:%S.%f').dt.time
    narrations['stop_seconds'] = pd.to_datetime(narrations['stop_timestamp'], format='%H:%M:%S.%f').dt.time
    narrations['narr_seconds'] = pd.to_datetime(narrations['narration_timestamp'], format='%H:%M:%S.%f').dt.time
    narrations['start_seconds'] = narrations['start_seconds'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)
    narrations['stop_seconds'] = narrations['stop_seconds'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)
    narrations['narr_seconds'] = narrations['narr_seconds'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)

    #get the narrations that fall within current considered segment
    narrations = narrations.sort_values('start_seconds')
    narrations_df = narrations[narrations.stop_seconds <=args.N].reset_index(drop=True)

    audio_events = [class_mapping[pred][0] for pred in unique_predictions]
    audio_events_df = pd.DataFrame({"description": audio_events, "start_seconds": start_times, "stop_seconds": end_times, "confidence": event_confs})
    audio_events_df = audio_events_df[audio_events_df.stop_seconds <=args.N]

    # Compute embeddings of narrations and audio event descriptions
    narration_embeddings = text_similarity_model.encode(narrations_df['narration'].tolist(), convert_to_tensor=True)#.cpu()
    audio_event_embeddings = text_similarity_model.encode(audio_events_df['description'].tolist(), convert_to_tensor=True)#.cpu()

    print("Generating audio-grounded narrations...")
    audio_event_to_narration = assign_audio_events(narrations_df, audio_events_df, narration_embeddings, audio_event_embeddings, text_similarity_model,
                                                    mode=args.audio_assignment_scheme, use_llama_assignment=args.use_llama_assignment,
                                                    audio_narr_sim_thresh=args.audio_narration_similarity_threshold,
                                                    tokenizer=tokenizer if args.use_llama_assignment else None,
                                                    llama3_model=llama3_model if args.use_llama_assignment else None)

    # Calculating and storing the union of assigned audio event intervals
    narration_segments = {idx: [] for idx in narrations_df.index}
    for ae_idx, (nar_idx, _) in audio_event_to_narration.items():
        if nar_idx is not None:
            narration_segments[nar_idx].append((audio_events_df.loc[ae_idx, 'start_seconds'], audio_events_df.loc[ae_idx, 'stop_seconds']))

    for idx in narration_segments:
        if args.multi_interval_assignment:
            sorted_intervals = sorted(narration_segments[idx], key=lambda x: x[0])
            merged_intervals = []
            for interval in sorted_intervals:
                if not merged_intervals or merged_intervals[-1][1] < interval[0]:
                    merged_intervals.append(list(interval))
                else:
                    merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])
            narration_segments[idx] = merged_intervals
        else:
            merged_intervals = []
            if narration_segments[idx]:
                start = min([interval[0] for interval in narration_segments[idx]])
                end = max([interval[1] for interval in narration_segments[idx]])
                merged_intervals.append((start, end))
            narration_segments[idx] = merged_intervals

    narrations_df['assigned_intervals'] = narrations_df.index.map(lambda x: narration_segments[x] if x in narration_segments else [])

    #save narrations to disk
    audio_grounded_narrs_output_dir = "audio_grounded_narrations"
    audio_grounded_narrs_filename = f"llama_assigned={args.use_llama_assignment}_clustered_bounds={args.use_clustered_bounds}_audio_assignment={args.audio_assignment_scheme}.pkl"
    if not os.path.exists(os.path.join("/private/home/arjunrs1/epic-sounds-annotations", audio_grounded_narrs_output_dir, video_id)):
        os.makedirs(os.path.join("/private/home/arjunrs1/epic-sounds-annotations", audio_grounded_narrs_output_dir, video_id))
    audio_grounded_narrations_filepath = os.path.join("/private/home/arjunrs1/epic-sounds-annotations",
                                                       audio_grounded_narrs_output_dir, video_id, audio_grounded_narrs_filename)
    narrations_df.to_pickle(audio_grounded_narrations_filepath)

    audio_predictions_output_dir = "audio_event_detection_predictions"
    audio_predictions_filename = f"dendrogram_height={args.dendrogram_height}.pkl"
    if not os.path.exists(os.path.join("/private/home/arjunrs1/epic-sounds-annotations", audio_predictions_output_dir, video_id)):
        os.makedirs(os.path.join("/private/home/arjunrs1/epic-sounds-annotations", audio_predictions_output_dir, video_id))
    audio_events_predictions_filepath = os.path.join("/private/home/arjunrs1/epic-sounds-annotations", 
                                                     audio_predictions_output_dir, video_id, audio_predictions_filename)
    audio_events_df.to_pickle(audio_events_predictions_filepath)

    """ 
    TODO: Need to fix the audio bounds generation -> Try different dendogram heights again.
    """

if __name__ == "__main__":
    main()