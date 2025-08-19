#!/bin/bash

# 初始化默认值（可选）
sft_model_ckpt_path="checkpoints/sft_llava"
IMAGE_FOLDER="dataset/CleanBench-Real_80k/images"
DATA_PATH="dataset/rrhf_train.json"
OFFLINE_DATA_PATH="dataset/offline_data.json"
OUTPUT="results/CheckpointsOut/trainv3"

while getopts "c:i:d:o:p:" opt; do
    case $opt in
        c) sft_model_ckpt_path="$OPTARG" ;;
        i) IMAGE_FOLDER="$OPTARG" ;;
        d) DATA_PATH="$OPTARG" ;;
        o) OFFLINE_DATA_PATH="$OPTARG" ;;
        p) OUTPUT="$OPTARG" ;;
        *) echo "usage: $0 -c <Model Path> -i <Image folder path> -d <Data files> -o <Offline data> -p <Output Path>"
           exit 1
    esac
done

if [ -z "$sft_model_ckpt_path" ] || [ -z "$IMAGE_FOLDER" ] || [ -z "$DATA_PATH" ] || [ -z "$OUTPUT" ]; then
    echo "Error: Required parameter missing!"
    exit 1
fi

echo "Model Path: $sft_model_ckpt_path"
echo "Image folder path: $IMAGE_FOLDER"
echo "Data files: $DATA_PATH"
echo "Offline data: $OFFLINE_DATA_PATH"
echo "Output Path: $OUTPUT"

CUDA_LIST=(0,1,2,3,4,5,6,7)
CUR_DIR=`pwd`
TEMPLATE=llama_3
MAX_GENERATION_LENGTH_OF_SAMPLING=384
actor_zero_stage=1
ACTOR_LEARNING_RATE=1e-5
CRITIC_LEARNING_RATE=2e-5
EPOCH=3 
TRAIN_SPLIT_RATIO=0.9997
DATA="llava_ppo"
DATA_SAMPLE="all"
IMAGE_PER_SAMPLE="1"


ROOT=${CUR_DIR}
export PYTHONPATH=${ROOT}:${PYTHONPATH}
mkdir -p $OUTPUT

CURRENT_PATH=$PATH
CONDA_PREFIX_PATH=$CONDA_PREFIX

deepspeed --include localhost:$(IFS=,; echo "${CUDA_LIST[*]}") --master_port 12346 src/mrrhf/ppo_main.py --max_seq_len 1472 --Offline_data_path ${OFFLINE_DATA_PATH} \
    --data_path ${DATA_PATH} --image_folder ${IMAGE_FOLDER} --template ${TEMPLATE} \
    --dataset_names ${DATA} --dataset_samples ${DATA_SAMPLE} --data_train_split_ratio ${TRAIN_SPLIT_RATIO} \
    --dataset_concatenate_samples ${IMAGE_PER_SAMPLE} --max_num_image_per_sample 8 \
    --gradient_checkpointing --model_architecture llava \
    --gradient_accumulation_steps 8 --num_warmup_steps 0.1 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --output_dir $OUTPUT  \
    --num_train_epochs ${EPOCH} \
    --lang_decoder_update --precision bf16 \
    --from_checkpoint $sft_model_ckpt_path \
    --actor_zero_stage $actor_zero_stage --offload_actor_model \
    --actor_learning_rate $ACTOR_LEARNING_RATE \
    --max_generation_length_of_sampling ${MAX_GENERATION_LENGTH_OF_SAMPLING} 

