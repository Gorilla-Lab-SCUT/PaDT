PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"
# Change the data_paths and image_folders to your own data
data_paths="PaDT-MLLM/ReferringImageCaptioning/ric_instances_train2017.json:PaDT-MLLM/COCO/instances_train2017.json:PaDT-MLLM/COCO/instances_train2017.json:PaDT-MLLM/RefCOCO/refcoco_train.json:PaDT-MLLM/RefCOCO/refcoco+_train.json:PaDT-MLLM/RefCOCO/refcocog_train.json"
image_folders="${REPO_HOME}/dataset/coco/train2017:${REPO_HOME}/dataset/coco/train2017:${REPO_HOME}/dataset/coco/train2017:${REPO_HOME}/dataset/coco/train2014:${REPO_HOME}/dataset/coco/train2014:${REPO_HOME}/dataset/coco/train2014"

model_path="Qwen/Qwen2.5-VL-3B-Instruct"
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"

export EXP_NAME="PaDT-Pro-3B" # TODO: change this to your own experiment name
cd ${REPO_HOME}/src/PaDT

# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
  sft_train.py \
    --output_dir ${REPO_HOME}/checkpoints/ours/sft/${EXP_NAME} \
    --resume_from_checkpoint true \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 4 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 100 \
    --learning_rate 2e-5 \
    --report_to wandb \
    --deepspeed ${REPO_HOME}/src/PaDT/local_scripts/zero3.json

echo "Training completed for ${EXP_NAME}"
