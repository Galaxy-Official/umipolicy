CUDA_VERSION="12.6"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-$CUDA_VERSION/lib64"
export PATH="/usr/local/cuda-$CUDA_VERSION/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-$CUDA_VERSION"
echo "$(nvcc -V)"

python -m scripts.flexiv_causal_inference \
    --debug_step 1000 \
    --pretrained_model_name_or_path "/Disk2/SnakeRob-DATA/checkpoints/dp_robopanoptes_sweep1_07-23/checkpoints/100000/pretrained_model/train_config.json" \
    --task_name "sweep-cube" \
    --video_type "camera" \
    --buffer_length 1 \
    --log_level "INFO" \
    --policy_type "diffusion" \
   
    # --pretrained_model_name_or_path "robot/checkpoints/rdt-170m-finetune-flexiv-hold-0402_2113/checkpoint-30000" \
    # --lang_embeddings_path "robot/embeddings/pick_apple.pt" \
    # --pretrained_model_name_or_path "robot/checkpoints/rdt-170m-finetune-flexiv-pour-water2-side-0326_2054/checkpoint-10500" \
    # --pretrained_model_name_or_path "robot/checkpoints/rdt-170m-finetune-flexiv-pour-water-side-0324_1123/checkpoint-7500" \
    # --pretrained_model_name_or_path "robot/checkpoints/rdt-170m-finetune-flexiv-pour-water2-0324_1845/checkpoint-6500" \