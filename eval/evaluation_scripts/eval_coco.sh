export CHECKPOINT='PaDT-MLLM/PaDT_Pro_3B'
export LOG_SUFFIX='padt_pro_3b'

# Multi-GPU run inference and save to log files: (pred_comp file records response sentences, pred_results file records structured outputs, i.g. bbox, mask, ...)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port="12369" --nproc_per_node=8 inference_coco.py $CHECKPOINT $LOG_SUFFIX

# Get results
python eval_coco.py $LOG_SUFFIX


