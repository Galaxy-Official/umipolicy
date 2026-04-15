NUM_WORKERS_PER_GPU=8
TOTAL_SHARDS=$(( NUM_WORKERS_PER_GPU * 2 ))

echo "Launching $NUM_WORKERS_PER_GPU extractions on GPU 0..."
for (( i=0; i<$NUM_WORKERS_PER_GPU; i++ )); do
    CUDA_VISIBLE_DEVICES=0 python wan_va/extract_latents.py \
        --dataset_path /Users/macbookpro/Desktop/handcap_simple_sorting_phone_409 \
        --ckpt_path ./ckpt/lingbot-va-base \
        --chunk_size 2 \
        --height 256 \
        --width 320 \
        --shard_id $i \
        --num_shards $TOTAL_SHARDS &
done

echo "Launching $NUM_WORKERS_PER_GPU extractions on GPU 1..."
for (( i=$NUM_WORKERS_PER_GPU; i<$TOTAL_SHARDS; i++ )); do
    CUDA_VISIBLE_DEVICES=1 python wan_va/extract_latents.py \
        --dataset_path /Users/macbookpro/Desktop/handcap_simple_sorting_phone_409 \
        --ckpt_path ./ckpt/lingbot-va-base \
        --chunk_size 2 \
        --height 256 \
        --width 320 \
        --shard_id $i \
        --num_shards $TOTAL_SHARDS &
done

echo "Running 16 parallel extractions on 2 H200 GPUs... Waiting for validation... (This will crush your CPUs/GPUs, please wait!)"
wait
echo "Extraction completed across all GPUs!"
