set -x 
export PYTHONPATH=/mmfs1/home/mickel7/code/OpenRLHF
export CUDA_HOME=/mmfs1/gscratch/h2lab/mickel/Miniconda3/envs/ray-rlhf
export LD_LIBRARY_PATH=/mmfs1/gscratch/h2lab/mickel/Miniconda3/envs/ray-rlhf/lib:$LD_LIBRARY_PATH
export PATH="$CUDA_HOME/bin:$PATH"

export CURRENT_DATETIME=$(date +"%m%d%y_%H%M%S")

python3 examples/train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --ref_gpu_type a100 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --reward_gpu_type a100 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --critic_gpu_type l40 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --actor_gpu_type l40 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_type a100 \
    --pretrain allenai/tulu-2-7b \
    --critic_pretrain openbmb/UltraRM-13b \
    --reward_pretrain openbmb/UltraRM-13b \
    --save_path ~/code/OpenRLHF/ckpt/tulu-2-7b-ppo-$CURRENT_DATETIME \
    --value_loss_coef 1.0 \
    --micro_train_batch_size 4 \
    --train_batch_size 64 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 64 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --rollout_temperature 0.7 \
    --zero_stage 2 \
    --actor_learning_rate 1e-6 \
    --critic_learning_rate 1e-6 \
    --init_kl_coef 0.05 \
    --prompt_data openbmb/UltraFeedback \
    --prompt_data_probs 1.0 \
    --actor_init_on_gpu \
    --gradient_checkpointing \
    --flash_attn \
    --bf16 \
    --save_steps 100 \
    --adam_offload \
    --pad_token "<unk>" \
    --warmup_ratio 0.10 \
    --value_head_name regression_head \
    --use_wandb $WANDB_API_KEY \
    --wandb_run_name tulu-2-7b-ppo-v1.34-DEBUG \
    # --seed 42 \
    # --normalize_reward \
    # argilla/ultrafeedback-binarized-preferences-cleaned
    