set -x 
# export PYTHONPATH=/mmfs1/home/mickel7/code/OpenRLHF
# export CUDA_HOME=/mmfs1/gscratch/h2lab/mickel/Miniconda3/envs/ray-rlhf
# export LD_LIBRARY_PATH=/mmfs1/gscratch/h2lab/mickel/Miniconda3/envs/ray-rlhf/lib:$LD_LIBRARY_PATH
# export PATH="$CUDA_HOME/bin:$PATH"

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
    --pretrain OpenLLMAI/Llama-2-7b-sft-model-ocra-500k \
    --reward_pretrain OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt \
    --save_path /openrlhf/examples/test_scripts/ckpt/7b_llama \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 512 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data Open-Orca/OpenOrca,Dahoas/full-hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward \
    --prompt_data_probs 0.4,0.5,0.1 \
    --max_samples 80000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --gradient_checkpointing \
    --flash_attn \
    --bf16 \
    --use_wandb ead86be5d8bada37193b2b3d6d7d9cc52aac151e \
    # --vllm_rtx6k \
    # --colocate_ref_reward \
    # --colocate_actor_critic \
    