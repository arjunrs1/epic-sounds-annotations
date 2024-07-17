#!/bin/bash

# Define the parameters
PID=$1
VID=$2

# Generate audio test set
python generate_audio_test_set.py --participant_id $PID --video_num $VID

export PYTHONPATH=/private/home/arjunrs1/epic-sounds-annotations/src:$PYTHONPATH

# Run the network
python ../tools/run_net.py \
--cfg ../configs/EPIC-Sounds/slowfast/SLOWFASTAUDIO_8x8_R50_TIM.yaml \
TRAIN.ENABLE False \
TEST.ENABLE True \
NUM_GPUS 8 \
OUTPUT_DIR ../outputs/${PID}_${VID}_win=1.0_hop=0.2 \
EPICSOUNDS.AUDIO_DATA_FILE /private/home/arjunrs1/EPIC-SOUNDS/processed_audios/EPIC-KITCHENS-100_audio.hdf5 \
EPICSOUNDS.ANNOTATIONS_DIR /private/home/arjunrs1/epic-sounds-annotations/EPIC_Sounds_train.csv \
TEST.CHECKPOINT_FILE_PATH /private/home/arjunrs1/EPIC-SOUNDS/asf_checkpoint/asf_epicsounds_1_sec.pyth \
EPICSOUNDS.TEST_LIST /private/home/arjunrs1/epic-sounds-annotations/per-video-annotations/${PID}_${VID}/EPIC_Sounds_recognition_${PID}_${VID}_win=1.0_hop=0.2_test_timestamps_tim_formatted.pkl

# Generate audio grounding
python generate_audio_grounding.py --participant_id $PID --video_num $VID

# Generate frame grounding
python generate_frame_grounding.py --participant_id $PID --video_num $VID

# Evaluate grounding
python evaluate_grounding.py --participant_id $PID --video_num $VID