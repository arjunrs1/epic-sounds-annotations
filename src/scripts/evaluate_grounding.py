import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import argparse
from ipywidgets import interact, IntSlider, fixed


def process_verb_iou(df):
    verbs = df['narration'].str.extract(r'\b(\w+)\b', expand=False).unique()
    verbs = np.delete(verbs, np.argwhere(verbs == "still"))  # Exclude non-verb words

    verb_iou = {}
    for verb in verbs:
        verb_df = df[df['narration'].str.contains(f"\\b{verb}\\b")]
        verb_iou[verb] = (len(verb_df), verb_df['IoU'].mean())

    verb_iou = dict(sorted(verb_iou.items(), key=lambda item: item[1][1], reverse=True))
    return pd.DataFrame.from_dict(verb_iou, orient='index', columns=['count', 'mean_IoU'])

def find_mean_duration(row, verb_durations_mapping, sub=False):
    for verb in verb_durations_mapping['verb_type']:
        if verb in row['narration']:
            mean_duration = verb_durations_mapping.loc[verb_durations_mapping['verb_type'] == verb, 'mean_duration'].values[0]
            if sub:
                return row['narr_seconds'] - mean_duration / 2
            else:
                return row['narr_seconds'] + mean_duration / 2
    return row['narr_seconds']  # Default if no verb is found

def update_plot(window_idx, vis_win_length, audio, sr, audio_grounded_narrations_df,
                 video_grounded_narrations_df, grounding, audio_events_df,
                 audio_annotations_gt):
    # Calculate the start and end time of the window
    start_time = window_idx * vis_win_length
    end_time = start_time + vis_win_length
    
    # Get the audio samples and corresponding timestamps for the window
    window_audio = audio[int(start_time * sr):int(end_time * sr)]
    window_timestamps = np.linspace(start_time, end_time, len(window_audio))
    
    # Plot the audio waveform
    fig, axs = plt.subplots(5, figsize=(15,15))
    axs[0].plot(window_timestamps, window_audio)
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Ground Truth Narration Intervals')
    
    #Plot the ground truth narration segments
    for _, row in audio_grounded_narrations_df.iterrows():
        if row['start_seconds'] >= start_time and row['stop_seconds'] <= end_time:
            axs[0].axvspan(row['start_seconds'], row['stop_seconds'], color='red', alpha=0.3)
            mid_point = (row['start_seconds'] + row['stop_seconds']) / 2
            axs[0].annotate(row['narration'],
                        (mid_point, 0),
                        textcoords="offset points",
                        xytext=(0, 40),  # Position text below the waveform
                        ha='center',
                        va='bottom',
                        rotation=0)  # Vertical text for clarity
                        
    # Plot the audio waveform
    axs[1].plot(window_timestamps, window_audio)
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Video-Assigned Narration Intervals (EgoVLP)')
    
    #Plot the ground truth narration segments
    for _, row in video_grounded_narrations_df.iterrows():
        start = row['assigned_intervals'][0][0]
        end = row['assigned_intervals'][0][1]
        if start >= start_time and end <= end_time:
            axs[1].axvspan(start, end, color='red', alpha=0.3)
            mid_point = (start + end) / 2
            axs[1].annotate(row['narration'],
                        (mid_point, 0),
                        textcoords="offset points",
                        xytext=(0, 40),  # Position text below the waveform
                        ha='center',
                        va='bottom',
                        rotation=0)  # Vertical text for clarity

    axs[2].plot(window_timestamps, window_audio)
    axs[2].set_ylabel('Amplitude')
    axs[2].set_title('Audio-Assigned Narration Intervals')
            
    # Overlay the predicted narration segments on the plot
    for interval_start, interval_stop, narration in grounding:
        if interval_start >= start_time and interval_stop <= end_time:
            axs[2].axvspan(interval_start, interval_stop, color='red', alpha=0.3)
            mid_point = (interval_start + interval_stop) / 2
            axs[2].annotate(narration,
                        (mid_point, 0),
                        textcoords="offset points",
                        xytext=(0, 40),  # Position text below the waveform
                        ha='center',
                        va='bottom',
                        rotation=0)  # Vertical text for clarity

    axs[3].plot(window_timestamps, window_audio)
    axs[3].set_ylabel('Amplitude')
    axs[3].set_title('Predicted Audio Events')
    
    #Plot the predicted audio events
    for _, row in audio_events_df.iterrows():
        if row['start_seconds'] >= start_time and row['stop_seconds'] <= end_time:
            axs[3].axvspan(row['start_seconds'], row['stop_seconds'], color='red', alpha=0.3)
            mid_point = (row['start_seconds'] + row['stop_seconds']) / 2
            axs[3].annotate(row['description'],
                        (mid_point, 0),
                        textcoords="offset points",
                        xytext=(0, 40),  # Position text below the waveform
                        ha='center',
                        va='bottom',
                        rotation=90)  # Vertical text for clarity

    axs[4].plot(window_timestamps, window_audio)
    axs[4].set_xlabel('Time (seconds)')
    axs[4].set_ylabel('Amplitude')
    axs[4].set_title('Ground Truth Audio Events')  

    #Plot the ground truth audio events
    for _, row in audio_annotations_gt.iterrows():
        if row['start_seconds'] >= start_time and row['stop_seconds'] <= end_time:
            axs[4].axvspan(row['start_seconds'], row['stop_seconds'], color='red', alpha=0.3)
            mid_point = (row['start_seconds'] + row['stop_seconds']) / 2
            axs[4].annotate(row['class'],
                        (mid_point, 0),
                        textcoords="offset points",
                        xytext=(0, 40),  # Position text below the waveform
                        ha='center',
                        va='bottom',
                        rotation=90)  # Vertical text for clarity
    
    fig.tight_layout()
    
def compute_union_intervals_and_max_duration(df, narration_col='narration', intervals_col='assigned_intervals'):
    grounding = []
    max_interval_narr = (None, 0)

    for _, row in df.iterrows():
        narration = row[narration_col]
        assigned_intervals = row[intervals_col]
        
        if assigned_intervals:
            min_start = float('inf')
            max_stop = float('-inf')
            
            for interval in assigned_intervals:
                interval_start, interval_stop = interval
                if interval_start < min_start:
                    min_start = interval_start
                if interval_stop > max_stop:
                    max_stop = interval_stop
            
            union_duration = max_stop - min_start
            grounding.append((min_start, max_stop, narration))
            
            if union_duration > max_interval_narr[1]:
                max_interval_narr = (narration, union_duration)

    return grounding, max_interval_narr

def calculate_audio_event_iou(segment1, segment2):
    start_max = max(segment1[0], segment2[0])
    end_min = min(segment1[1], segment2[1])
    intersection = max(0, end_min - start_max)
    union = (segment1[1] - segment1[0]) + (segment2[1] - segment2[0]) - intersection
    return intersection / union if union != 0 else 0

def interval_intersection(A, B):
    """ Find the intersection of two interval lists """
    i, j, intersection = 0, 0, []
    while i < len(A) and j < len(B):
        a_start, a_end = A[i]
        b_start, b_end = B[j]
        # Find the intersection range
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start <= end:  # There is an overlap
            intersection.append((start, end))
        # Move to next interval in A or B
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return intersection

def merge_intervals(intervals):
    """ Merge overlapping intervals into a single list of intervals """
    if not intervals:
        return []
    # Sort intervals based on start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        # If current interval overlaps with the last one, merge them
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged

def compute_iou(row, mode=None):
    """ Compute the IoU based on ground truth and assigned intervals """
    ground_truth_intervals = [(row['start_seconds'], row['stop_seconds'])]
    if mode=="baseline":
        assigned_intervals = [[row['baseline_start'], row['baseline_stop']]]
    if mode=="duration baseline":
        assigned_intervals = [[row['duration_baseline_start'], row['duration_baseline_stop']]]
    elif mode is None:
        assigned_intervals = row['assigned_intervals']
    if not assigned_intervals:
        return 0  # No IoU if there are no assigned intervals
    # Merge intervals to find the union
    union_intervals = merge_intervals(ground_truth_intervals + assigned_intervals)
    intersection_intervals = interval_intersection(ground_truth_intervals, assigned_intervals)
    # Calculate areas
    intersection_area = sum(end - start for start, end in intersection_intervals)
    union_area = sum(end - start for start, end in union_intervals)
    if union_area == 0:
        return 0  # Avoid division by zero
    return intersection_area / union_area

def main():
    parser = argparse.ArgumentParser(description="Process audio and video annotations with various grounding and evaluation parameters.")
    
    # File-loading options
    parser.add_argument('--participant_id', type=str, help='Participant ID')
    parser.add_argument('--video_num', type=str, help='Video number')
    parser.add_argument('--sr', type=int, default=24000, help='Audio sampling rate')
    parser.add_argument('--audio_annotations_train_file', type=str, default="/private/home/arjunrs1/epic-sounds-annotations/EPIC_Sounds_train.pkl", help='Path to the training audio annotations file')
    parser.add_argument('--audio_annotations_val_file', type=str, default="/private/home/arjunrs1/epic-sounds-annotations/EPIC_Sounds_validation.pkl", help='Path to the validation audio annotations file')
    
    # Options
    parser.add_argument('--N', type=int, default=None, help='Number of seconds to trim the audio')
    parser.add_argument('--best_iou_threshold', type=float, default=0.5, help='Threshold for best IOU for audio event detection evaluation')
    parser.add_argument('--use_clustered_bounds', action='store_true', help='Use clustered bounds for audio event detection')
    parser.add_argument('--dendrogram_height', type=float, default=0.8, help='Dendrogram height for clustering')
    parser.add_argument('--use_llama_assignment', action='store_true', help='Use Llama model for assignment')
    parser.add_argument('--audio_assignment_scheme', type=str, default="best", choices=['best', 'random'], help='Scheme for assigning audio to narrations')
    parser.add_argument('--similarity_threshold', type=float, default=0.3, help='Similarity threshold for video grounding')
    parser.add_argument('--feature_stride', type=float, default=0.5, help='Feature stride for video grounding')
    parser.add_argument('--feature_duration', type=float, default=1.0, help='Feature duration for video grounding')
    parser.add_argument('--num_unique_narrs_for_disp', type=int, default=20, help='Number of unique narrations for display')
    parser.add_argument('--vis_win_length', type=int, default=20, help='Window length for visualization')

    args = parser.parse_args()

    # Construct video_id using provided arguments
    video_id = f"{args.participant_id}_{args.video_num}"

    audio_path = f'/private/home/arjunrs1/EPIC-SOUNDS/{video_id}.wav'
    audio, sr = librosa.load(audio_path, sr=args.sr)
    if args.N is None:
        args.N = int(len(audio) / args.sr)
    N_seconds_samples = args.N * args.sr
    audio = audio[:N_seconds_samples]

    # Load annotations
    audio_annotations_train = pd.read_pickle(args.audio_annotations_train_file)
    audio_annotations_val = pd.read_pickle(args.audio_annotations_val_file)
    audio_annotations = pd.concat([audio_annotations_train, audio_annotations_val], axis=0)

    audio_annotations_gt = audio_annotations[audio_annotations.video_id==video_id]
    audio_annotations_gt = audio_annotations_gt[['start_timestamp', 'stop_timestamp', 'description', 'class', 'class_id']].copy()
    audio_annotations_gt['start_seconds'] = pd.to_datetime(audio_annotations_gt['start_timestamp'], format='%H:%M:%S.%f').dt.time
    audio_annotations_gt['stop_seconds'] = pd.to_datetime(audio_annotations_gt['stop_timestamp'], format='%H:%M:%S.%f').dt.time
    audio_annotations_gt['start_seconds'] = audio_annotations_gt['start_seconds'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)
    audio_annotations_gt['stop_seconds'] = audio_annotations_gt['stop_seconds'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)

    #--------------------------------------------------------load in predicted audio events:-------------------------------------------------------
    with open(f"/private/home/arjunrs1/epic-sounds-annotations/audio_event_detection_predictions/{video_id}/dendrogram_height={args.dendrogram_height}.pkl", "rb") as f:
        audio_events_df = pickle.load(f)

    #--------------------------------------------------------load in predicted audio grounding:-------------------------------------------------------
    #Load in narrations_df here from generate_audio_grounding.py script output file
    with open(f"/private/home/arjunrs1/epic-sounds-annotations/audio_grounded_narrations/{video_id}/llama_assigned={args.use_llama_assignment}_clustered_bounds={args.use_clustered_bounds}_audio_assignment={args.audio_assignment_scheme}.pkl", "rb") as f:
        audio_grounded_narrations_df = pickle.load(f)

    #--------------------------------------------------------load in predicted video grounding:-------------------------------------------------------
    #Load in narrations_df here from generate_frame_grounding.py script output file
    with open(f"/private/home/arjunrs1/epic-sounds-annotations/video_grounded_narrations/{video_id}/similarity_threshold={args.similarity_threshold}_feature_duration={args.feature_duration}_feature_stride={args.feature_stride}.pkl", "rb") as f:
        video_grounded_narrations_df = pickle.load(f)

    #--------------------------------------------------------evaluate audio event merging against GT:-------------------------------------------------------
    matches = []
    for index, pred in audio_events_df.iterrows():
        best_iou = 0
        best_gt = None
        for _, gt in audio_annotations_gt.iterrows():
            if pred['description'] == gt['class']:
                iou = calculate_audio_event_iou((pred['start_seconds'], pred['stop_seconds']), (gt['start_seconds'], gt['stop_seconds']))
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
        if best_iou > args.best_iou_threshold:
            matches.append((pred, best_gt, best_iou))

    # Convert matches to DataFrame if needed for further analysis
    matches_df = pd.DataFrame(matches, columns=['Prediction', 'Ground Truth', 'IoU'])

    tp = len(matches_df)
    fp = len(audio_events_df) - tp
    fn = len(audio_annotations_gt) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n")
    print("Audio Event Detection Metrics:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print("\n")

    #fewer segments leads to higher precision but lower recall (and lower f1 score). Which do we care more about for our purposes?
    ### Format DataFrames for analysis:

    #--------------------------------------------------compute IoU for predicted grounded narrations:-------------------------------------------------------
    # Apply IoU computation to the dataframe
    audio_grounded_narrations_df['IoU'] = audio_grounded_narrations_df.apply(compute_iou, axis=1)
    video_grounded_narrations_df['IoU'] = video_grounded_narrations_df.apply(compute_iou, axis=1)

    #-----------------compute naive baseline (uniform) and filter narrations to only those which received an audio event-------------------------------------------------------
    # Number of narrations
    num_narrations = len(audio_grounded_narrations_df)

    # Compute segment length
    segment_length = args.N / num_narrations
    mean_gt_narration_duration = audio_grounded_narrations_df.sort_values(by='narr_seconds')['narr_seconds'].diff().shift(-1).iloc[:-1].mean()
    print("Mean GT Narration Duration:")
    print(mean_gt_narration_duration)
    print("\n")

    verb_durations_mapping = pd.read_csv('/private/home/arjunrs1/epic-kitchens-100-annotations/train_narration_verb_duration.csv')

    # Assign segments to narrations
    audio_grounded_narrations_df['baseline_start'] = audio_grounded_narrations_df['narr_seconds'] - mean_gt_narration_duration / 2
    audio_grounded_narrations_df['baseline_stop'] = audio_grounded_narrations_df['narr_seconds'] + mean_gt_narration_duration / 2

    # Apply the function to compute duration_baseline_start
    audio_grounded_narrations_df['duration_baseline_start'] = audio_grounded_narrations_df.apply(find_mean_duration,
                                                                                                  axis=1, 
                                                                                                  verb_durations_mapping=verb_durations_mapping,
                                                                                                    sub=True)
    audio_grounded_narrations_df['duration_baseline_stop'] = audio_grounded_narrations_df.apply(find_mean_duration,
                                                                                                 axis=1, 
                                                                                                 verb_durations_mapping=verb_durations_mapping)

    # Apply IoU calculation to the dataframe for baseline
    audio_grounded_narrations_df['baseline_IoU'] = audio_grounded_narrations_df.apply(compute_iou, mode="baseline", axis=1)
    audio_grounded_narrations_df['duration_baseline_IoU'] = audio_grounded_narrations_df.apply(compute_iou, mode="duration baseline", axis=1)

    # Filter out rows where 'assigned_intervals' is not empty (i.e., no audio events were assigned to that narration)
    filtered_narrations = audio_grounded_narrations_df[audio_grounded_narrations_df['assigned_intervals'].apply(bool)]
    video_grounded_narrations_filtered_df = video_grounded_narrations_df.loc[filtered_narrations.index]
    ### IoU Metrics:

    #------------------------------compute IoU @ thresholds for our method and baselines on filtered and full set of narrations:-------------------------------------------------------
    # Define IoU thresholds
    iou_thresholds = [0.1, 0.3, 0.5]

    # Metrics for unfiltered audio-assignment
    audio_assignment_metrics = {
        'Mean IoU': audio_grounded_narrations_df['IoU'].mean(),
        **{f'IoU >= {threshold}': (
            audio_grounded_narrations_df[audio_grounded_narrations_df['IoU'] >=
                                          threshold].shape[0] / audio_grounded_narrations_df.shape[0]
                                            * 100) for threshold in iou_thresholds}
    }

    # Metrics for filtered audio-assignment
    audio_assignment_filtered_metrics = {
        'Mean IoU': filtered_narrations['IoU'].mean(),
        **{f'IoU >= {threshold}': (
            filtered_narrations[filtered_narrations['IoU'] >=
                                 threshold].shape[0] / filtered_narrations.shape[0]
                                   * 100) for threshold in iou_thresholds}
    }

    # Metrics for unfiltered video-assignment
    video_assignment_metrics = {
        'Mean IoU': video_grounded_narrations_df['IoU'].mean(),
        **{f'IoU >= {threshold}': (
            video_grounded_narrations_df[video_grounded_narrations_df['IoU'] >=
                                          threshold].shape[0] / video_grounded_narrations_df.shape[0]
                                            * 100) for threshold in iou_thresholds}
    }

    # Metrics for filtered video-assignment
    video_assignment_filtered_metrics = {
        'Mean IoU': video_grounded_narrations_filtered_df['IoU'].mean(),
        **{f'IoU >= {threshold}': (
            video_grounded_narrations_filtered_df[video_grounded_narrations_filtered_df['IoU'] >=
                                                   threshold].shape[0] / video_grounded_narrations_filtered_df.shape[0]
                                                     * 100) for threshold in iou_thresholds}
    }

    # Metrics for baseline (unfiltered)
    baseline_metrics = {
        'Mean IoU': audio_grounded_narrations_df['baseline_IoU'].mean(),
        **{f'IoU >= {threshold}': (
            audio_grounded_narrations_df[audio_grounded_narrations_df['baseline_IoU'] >=
                                          threshold].shape[0] / audio_grounded_narrations_df.shape[0]
                                            * 100) for threshold in iou_thresholds}
    }

    # Metrics for baseline (filtered)
    baseline_filtered_metrics = {
        'Mean IoU': filtered_narrations['baseline_IoU'].mean(),
        **{f'IoU >= {threshold}': (
            filtered_narrations[filtered_narrations['baseline_IoU'] >= 
                                threshold].shape[0] / filtered_narrations.shape[0]
                                  * 100) for threshold in iou_thresholds}
    }

    # Metrics for duration baseline (unfiltered)
    duration_baseline_metrics = {
        'Mean IoU': audio_grounded_narrations_df['duration_baseline_IoU'].mean(),
        **{f'IoU >= {threshold}': (
            audio_grounded_narrations_df[audio_grounded_narrations_df['duration_baseline_IoU'] >=
                                          threshold].shape[0] / audio_grounded_narrations_df.shape[0]
                                            * 100) for threshold in iou_thresholds}
    }

    # Metrics for duration baseline (filtered)
    duration_baseline_filtered_metrics = {
        'Mean IoU': filtered_narrations['duration_baseline_IoU'].mean(),
        **{f'IoU >= {threshold}': (
            filtered_narrations[filtered_narrations['duration_baseline_IoU'] >=
                                 threshold].shape[0] / filtered_narrations.shape[0]
                                   * 100) for threshold in iou_thresholds}
    }

    # Compile into DataFrame
    metrics_df = pd.DataFrame({
        'Method': ['Audio-Assignment',
                    'Audio-Assignment (filtered)',
                      'Video-Assignment', 
                      'Video-Assignment (filtered)', 
                      'Naive Baseline', 
                      '(Filtered)', 
                      'Narr Duration Baseline', 
                      '(Filtered)'],
        **{key: [audio_assignment_metrics[key], 
                 audio_assignment_filtered_metrics[key], 
                 video_assignment_metrics[key], 
                 video_assignment_filtered_metrics[key], 
                 baseline_metrics[key], 
                 baseline_filtered_metrics[key], 
                 duration_baseline_metrics[key], 
                 duration_baseline_filtered_metrics[key]] for key in audio_assignment_metrics}
    })

    # Set 'Method' as index
    metrics_df.set_index('Method', inplace=True)

    print("Narration Grounding Metrics:")
    print(metrics_df)

if __name__ == "__main__":
    main()