# Download VLM-R1 rec_jsons_processed, for the final results using VLM-R1 validation set.
cd ../../dataset/RefCOCO
if [ ! -d rec_jsons_processed ]; then
    wget https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/rec_jsons_processed.zip
    unzip rec_jsons_processed.zip
    rm rec_jsons_processed.zip
fi;
cd ../../eval/evaluation_scripts

# Inference Start
export CHECKPOINT='PaDT-MLLM/PaDT_Pro_3B'
export LOG_SUFFIX='padt_pro_3b'

for SPLIT in refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test
do
    echo 'Inferring' $SPLIT;
    # Multi-GPU run inference and save to log files: (pred_comp file records response sentences, pred_results file records structured outputs, i.g. bbox, mask, ...)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port="12369" --nproc_per_node=8 inference_refcoco.py $CHECKPOINT $SPLIT $LOG_SUFFIX

    # Get results
    python eval_refcoco.py $LOG_SUFFIX $SPLIT
    echo -e '\n\n';
done;


