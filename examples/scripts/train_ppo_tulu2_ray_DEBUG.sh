set -x 
REPO_PATH="/code/OpenRLHF"
export PYTHONPATH=$REPO_PATH
# export CUDA_HOME=/root/Miniconda3/condabin/conda/envs/openrlhf
# export LD_LIBRARY_PATH=/root/Miniconda3/condabin/conda/envs/openrlhf/lib:$LD_LIBRARY_PATH
# export PATH="$CUDA_HOME/bin:$PATH"

export CURRENT_DATETIME=$(date +"%m%d%y_%H%M%S")

python3 examples/train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --pretrain $REPO_PATH/resources/models/tulu-2-7b \
    --critic_pretrain $REPO_PATH/resources/models/UltraRM-13b \
    --reward_pretrain $REPO_PATH/resources/models/UltraRM-13b \
    --save_path $REPO_PATH/ckpt/tulu-2-7b-ppo-DEBUG-$CURRENT_DATETIME \
    --value_loss_coef 1.0 \
    --micro_train_batch_size 8 \
    --train_batch_size 8 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 8 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 512 \
    --rollout_temperature 0.7 \
    --zero_stage 2 \
    --actor_learning_rate 1e-6 \
    --critic_learning_rate 1e-6 \
    --init_kl_coef 0.05 \
    --prompt_data $REPO_PATH/resources/datasets/ultrafeedback-binarized-preferences-cleaned \
    --prompt_data_probs 1.0 \
    --actor_init_on_gpu \
    --gradient_checkpointing \
    --flash_attn \
    --bf16 \
    --save_steps -1 \
    --adam_offload \
    --pad_token "<unk>" \
    --warmup_ratio 0.05 \
    --head_prefix regression_head \
    --colocate_actor_ref \
    --use_wandb $WANDB_API_KEY \
    --wandb_run_name tulu-2-7b-ppo-DEBUG \
    --perf
    # --ultrarm_shift_template
    # --normalize_reward \