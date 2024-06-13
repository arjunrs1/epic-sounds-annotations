import numpy as np
import os
import random
import pickle
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.stats import mode
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def hard_merge_and_predict(predictions, window_size, hop_length, sampling_rate=24000):
    unique_predictions = []
    start_times = []
    end_times = []

    current_pred = predictions[0]
    start_time = 0

    for i, pred in enumerate(predictions[1:], start=1):
        if pred != current_pred:
            unique_predictions.append(current_pred)
            start_times.append(start_time)
            end_times.append((i - 1)*hop_length + window_size)  # Adding window duration to last index
            current_pred = pred
            start_time = i*hop_length

    # Adding the last segment
    unique_predictions.append(current_pred)
    start_times.append(start_time)
    end_times.append((len(predictions) - 1)*hop_length + window_size)

    return unique_predictions, start_times, end_times

def llama_audio_narration_assignment(narrations, audio_event):
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
    if "none" in parsed_answer.lower():
        parsed_answer = None
    return output, parsed_answer

def find_closest_narrations(audio_event_row, narrations_df):
    before_narrations = narrations_df[narrations_df['stop_seconds'] <= audio_event_row['start_seconds']]
    after_narrations = narrations_df[narrations_df['start_seconds'] >= audio_event_row['stop_seconds']]

    closest_before = before_narrations.iloc[-1] if not before_narrations.empty else None
    closest_after = after_narrations.iloc[0] if not after_narrations.empty else None
    return closest_before, closest_after

def assign_audio_events(narrations_df, audio_events_df, narr_embeds, audio_embeds, mode='best'):
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
            full_output, best_narration = llama_audio_narration_assignment([narr[0] for narr in llama_narrations], audio_event_row.description)
            if best_narration is not None:
                #embed the best narration, compare to each of the narration embeddings, and store overlaps as (nar_idx, similarity)
                llama_generated_narr = text_similarity_model.encode([best_narration], convert_to_tensor=True)#.cpu()
                for narr, nar_idx in llama_narrations:
                    similarity = cosine_similarity(narr_embeds[nar_idx].reshape(1, -1), llama_generated_narr.reshape(1, -1))[0][0]
                    overlaps.append((nar_idx, similarity))
                
        if overlaps:
            if mode=='best':
                best_narration = max(overlaps, key=lambda x: x[1])
                if best_narration[1] < audio_narration_similarity_threshold:
                    best_narration = (None, None)
            elif mode=='random':
                best_narration = random.choice(overlaps)
                if best_narration[1] < audio_narration_similarity_threshold:
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
                full_output, best_narration = llama_audio_narration_assignment([narr[0] for narr in llama_closest], audio_event_row.description)
                if best_narration is not None:
                    llama_generated_narr = text_similarity_model.encode([best_narration], convert_to_tensor=True)#.cpu()
                    for narr, nar_idx in llama_closest:
                        similarity = cosine_similarity(narr_embeds[nar_idx].reshape(1, -1), llama_generated_narr.reshape(1, -1))[0][0]
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
                    if best_narration[1] < audio_narration_similarity_threshold:
                        best_narration = (None, None)
                elif mode=='random':
                    best_narration = random.choice(closest_narrations)
                    if best_narration[1] < audio_narration_similarity_threshold:
                        best_narration = (None, None)
                assignment[audio_idx] = best_narration

    return assignment

#load in llama and sBERT models
tokenizer = AutoTokenizer.from_pretrained("/large_experiments/ram/shared/Meta-Llama-3-8B-Instruct-hf")
llama3_model = AutoModelForCausalLM.from_pretrained("/large_experiments/ram/shared/Meta-Llama-3-8B-Instruct-hf", device_map="cuda")
text_similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda")

#---------------------------------------------------------------------------------------FILE-LOADING OPTIONS--------------------------------------------------------
output_dir = "/private/home/arjunrs1/epic-sounds-annotations/src/outputs"
audio_annotations_train_file = "/private/home/arjunrs1/epic-sounds-annotations/EPIC_Sounds_train.pkl"
audio_annotations_val_file = "/private/home/arjunrs1/epic-sounds-annotations/EPIC_Sounds_validation.pkl"
class_mapping_file = '/private/home/arjunrs1/epic-sounds-annotations/annotation_mapping.pkl'
participant_id = "P01"
video_num = "01"
video_id = f"{participant_id}_{video_num}"
#Specify desired window length and stride of audio features to load in
win_length = 1.0 #options: 1.0, 2.0
hop_length = 0.2 #options: .2, 1.0
tim_formatted = "_tim_formatted" if win_length == 1.0 else ""
#---------------------------------------------------------------------------------------OTHER OPTIONS----------------------------------------------------------------
# Flag for how to compute audio events + bounds (clustering on class-confidence vs. max-confidence merging)
use_clustered_bounds = False
# Threshold similarity for an audio event to be assigned to a narration #Can be set to higher  if using use_llama_assignment
audio_narration_similarity_threshold = 0.4
# Height of dendrogram at which to cut (experiment with this) (only used if use_clustered_bounds is True)
dendrogram_height = 0.8
#Whether to use llama-based audio-narration assignment or direct sBERT text similarity
use_llama_assignment = False
# Trim the audio to desired number of seconds
N = 1620
# Flag for how to merge assigned intervals (multiple disjoint intervals per narration, or single interval)
multi_interval_assignment = True
# Flag for how to assign audio to narrations: "best": highest similarity, "random": random assignment (baseline)
audio_assignment_scheme = "best"
#options: {val, test} NOTE: test cannot be analyzed as we don't have GT - it is for the EPIC-KITCHENS challenge.
output_file = f"scores/EPIC_Sounds_recognition_{video_id}_win={win_length}_hop={hop_length}_test_timestamps{tim_formatted}.pkl"
#---------------------------------------------------------------------------------------OTHER OPTIONS----------------------------------------------------------------

#load in predictions and labels
with open(os.path.join(output_dir, f"{video_id}_win={win_length}_hop={hop_length}", output_file), 'rb') as f:
    pred_annotations = pickle.load(f)
probabilities = softmax(pred_annotations['interaction_output'], axis=1)
pred_labels = np.argsort(-probabilities, axis=1)[:, :1]

# Load audio file
audio_path = f'/private/home/arjunrs1/EPIC-SOUNDS/{participant_id}_{video_num}.wav'
audio, sr = librosa.load(audio_path, sr=None)
N_seconds_samples = N * sr
audio = audio[:N_seconds_samples]

# Load class predictions
predictions=pred_labels.flatten()[:int((N-win_length)/hop_length)+1]  # Load and trim
# Load and truncate class probabilities
probabilities=probabilities[:int((N-win_length)/hop_length)+1,:]  # Load and trim
# Load class mapping
with open(class_mapping_file, 'rb') as file:
    class_mapping = pickle.load(file)

#load in audio ground truth annotations:
audio_annotations_train = pd.read_pickle(audio_annotations_train_file)
audio_annotations_val = pd.read_pickle(audio_annotations_val_file)
audio_annotations = pd.concat([audio_annotations_train, audio_annotations_val], axis=0)

#Process annotations for target video:
audio_annotations_gt = audio_annotations[audio_annotations.video_id==video_id]
audio_annotations_gt = audio_annotations_gt[['start_timestamp', 'stop_timestamp', 'description', 'class', 'class_id']].copy()
audio_annotations_gt['start_seconds'] = pd.to_datetime(audio_annotations_gt['start_timestamp'], format='%H:%M:%S.%f').dt.time
audio_annotations_gt['stop_seconds'] = pd.to_datetime(audio_annotations_gt['stop_timestamp'], format='%H:%M:%S.%f').dt.time
audio_annotations_gt['start_seconds'] = audio_annotations_gt['start_seconds'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)
audio_annotations_gt['stop_seconds'] = audio_annotations_gt['stop_seconds'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)

#Compute audio events and their temporal bounds
if use_clustered_bounds:
    unique_predictions, start_times, end_times = cluster_and_predict(probabilities, window_size=win_length, hop_length=hop_length, dendrogram_height=dendrogram_height)
else:
    # Identify contiguous audio segments and calculate their intervals (get audio events and their grounding intervals)
    unique_predictions, start_times, end_times = hard_merge_and_predict(predictions, window_size=win_length, hop_length=hop_length)

#Load in narrations and ground truth bounds
narrations_df = pd.read_csv('~/epic-kitchens-100-annotations/EPIC_100_train.csv')
narrations_df = narrations_df[(narrations_df['participant_id'] == participant_id) & (narrations_df['video_id'] == video_id)]
narrations = narrations_df[['start_timestamp', 'stop_timestamp', 'narration']].copy()
narrations['start_seconds'] = pd.to_datetime(narrations['start_timestamp'], format='%H:%M:%S.%f').dt.time
narrations['stop_seconds'] = pd.to_datetime(narrations['stop_timestamp'], format='%H:%M:%S.%f').dt.time
narrations['start_seconds'] = narrations['start_seconds'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)
narrations['stop_seconds'] = narrations['stop_seconds'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)

#get the narrations that fall within current considered segment
narrations = narrations.sort_values('start_seconds')
narrations_df = narrations[narrations.stop_seconds <=N].reset_index(drop=True)

audio_events = [class_mapping[pred][0] for pred in unique_predictions]
audio_events_df = pd.DataFrame({"description": audio_events, "start_seconds": start_times, "stop_seconds": end_times})
audio_events_df = audio_events_df[audio_events_df.stop_seconds <=N]

# Compute embeddings of narrations and audio event descriptions
narration_embeddings = text_similarity_model.encode(narrations_df['narration'].tolist(), convert_to_tensor=True)#.cpu()
audio_event_embeddings = text_similarity_model.encode(audio_events_df['description'].tolist(), convert_to_tensor=True)#.cpu()

print("Generating audio-grounded narrations...")
audio_event_to_narration = assign_audio_events(narrations_df, audio_events_df, narration_embeddings, audio_event_embeddings, mode=audio_assignment_scheme)

# Calculating and storing the union of assigned audio event intervals
narration_segments = {idx: [] for idx in narrations_df.index}
for ae_idx, (nar_idx, _) in audio_event_to_narration.items():
    if nar_idx is not None:
        narration_segments[nar_idx].append((audio_events_df.loc[ae_idx, 'start_seconds'], audio_events_df.loc[ae_idx, 'stop_seconds']))

for idx in narration_segments:
    if multi_interval_assignment:
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
audio_grounded_narrs_filename = f"llama_assigned={use_llama_assignment}_clustered_bounds={use_clustered_bounds}_audio_assignment={audio_assignment_scheme}.pkl"
audio_grounded_narrations_filepath = os.path.join("/private/home/arjunrs1/epic-sounds-annotations", audio_grounded_narrs_output_dir, audio_grounded_narrs_filename)
narrations_df.to_pickle(audio_grounded_narrations_filepath)

audio_predictions_output_dir = "audio_event_detection_predictions"
audio_predictions_filename = f"dendrogram_height={dendrogram_height}.pkl"
audio_events_predictions_filepath = os.path.join("/private/home/arjunrs1/epic-sounds-annotations", audio_predictions_output_dir, audio_predictions_filename)
audio_events_df.to_pickle(audio_events_predictions_filepath)

""" 
TODO: Need to fix the audio bounds generation -> Try different dendogram heights again.
TODO: Look at TODOs in evaluate_grounding.ipynb -> There is an issue that we are selecting the last interval assigned to a narration (via audio)
as the ONLY interval for that narration.
TODO: Suppress low confidence detections heuristically (if confidence of max class is below a threshold, remove the prediction)
 """