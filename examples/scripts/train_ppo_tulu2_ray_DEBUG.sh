set -x 
export PYTHONPATH=/mmfs1/home/mickel7/code/OpenRLHF
export CUDA_HOME=/mmfs1/gscratch/h2lab/mickel/Miniconda3/envs/ray-rlhf
export LD_LIBRARY_PATH=/mmfs1/gscratch/h2lab/mickel/Miniconda3/envs/ray-rlhf/lib:$LD_LIBRARY_PATH
export PATH="$CUDA_HOME/bin:$PATH"

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
    --vllm_tensor_parallel_size 2 \
    --pretrain allenai/tulu-2-7b \
    --reward_pretrain openbmb/UltraRM-13b \
    --save_path ~/code/OpenRLHF/examples/test_scripts/ckpt/tulu-2-7b-ppo \
    --micro_train_batch_size 4 \
    --train_batch_size 16 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 16 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --rollout_temperature 1.0 \
    --zero_stage 2 \
    --actor_learning_rate 1e-6 \
    --critic_learning_rate 1e-6 \
    --init_kl_coef 0.05 \
    --prompt_data argilla/ultrafeedback-binarized-preferences-cleaned \
    --prompt_data_probs 1.0 \
    --max_samples 60000 \
    --actor_init_on_gpu \
    --adam_offload \
    --gradient_checkpointing \
    --flash_attn \
    --bf16 \
    --save_steps 1 \
    --vllm_separate_node \
    --normalize_reward \
    # --use_wandb $WANDB_API_KEY \
    # --wandb_run_name tulu-2-7b-ppo-v1.2 \
    # --colocate_actor_critic \
    # --vllm_rtx6k \
    # --colocate_ref_reward \
    