import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_reward, masked_mean
from openrlhf.utils.logging import init_logger
from openrlhf.utils.utils import debug_here

logger = init_logger(__name__)


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.advantages = self.advantages.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        self.values = self.values.pin_memory()
        self.returns = self.returns.pin_memory()
        self.advantages = self.advantages.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        num_actions = action_mask.size(1)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        value = self.critic(sequences, action_mask, attention_mask)

        # rewards
        r = self.reward_model(sequences, attention_mask)

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )
        
        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # reset model state
        self.actor.train()
        self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        
    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **kwargs) -> Experience:
        self.actor.eval()
        device = torch.cuda.current_device()
    
        # generate sequence
        start = time.time()
        (sequences, attention_mask, action_mask, eos_penalty_flags), (rm_sequences, rm_attention_mask, rm_action_mask, _) = (
            self._generate_local(prompts, **kwargs)
            if self.vllm_engines is None
            else self._generate_vllm(prompts, **kwargs)
        )
        generate_time = time.time() - start

        num_actions = action_mask.size(1)
        sequences_cpu, attention_mask_cpu, action_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
            action_mask.to("cpu"),
        )

        rm_sequences_cpu, rm_attention_mask_cpu, rm_action_mask_cpu = (
            rm_sequences.to("cpu"),
            rm_attention_mask.to("cpu"),
            rm_action_mask.to("cpu"),
        )

        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(sequences_cpu, num_actions, attention_mask_cpu)

        # values
        value_ref = self.critic.forward.remote(
            sequences_cpu, 
            action_mask_cpu, 
            attention_mask_cpu)

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward:
            ray.get([value_ref])

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])

        # rewards
        r_refs = []
        for rm in self.reward_model:
            r_refs.append(rm.forward.remote(rm_sequences_cpu, rm_attention_mask_cpu))

        # log probs
        start = time.time()
        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        actor_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        base_action_log_probs, value = base_action_log_probs.to(device), value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]
        print(f'eos_penalty ---> {kwargs.get("eos_penalty", False)}')
        if kwargs.get("eos_penalty", False):
            eos_penalty_flags = torch.tensor(eos_penalty_flags, device=device)
            r = torch.where(eos_penalty_flags, -10.0 * eos_penalty_flags.float(), r)

        total_reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            total_reward,
            action_mask,
            kwargs["gamma"],
            kwargs["lambd"],
        )

        info = {
            "kl_per_token": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "kl": (kl * action_mask).sum(dim=-1),
            "return": total_reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
            "prewhitten_advantage": masked_mean(advantage, action_mask, dim=-1),
        }
        # print(f"eos_reward: {info['reward']}")
        # print(f"kl: {info['kl']}")
        # print(f"padded_seq_len_policy_vs_rm: {sequences_cpu.shape[1]}, {rm_sequences_cpu.shape[1]}")
        if self.strategy.args.perf:
            batch_size = 1 if isinstance(prompts, str) else len(prompts)
            info["generate_time"] = torch.full((batch_size,), generate_time, device=device)
            info["actor_time"] = torch.full((batch_size,), actor_time, device=device)
            info["wait_time"] = torch.full((batch_size,), wait_time, device=device)

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

        # send experience to critic
        experience_cpu = deepcopy(experience)
        experience_cpu.to_device("cpu")
        self._ref = self.critic.append.remote(experience_cpu)

        # critic_experience = Experience(
        #     rm_sequences,
        #     action_log_probs,
        #     value,
        #     returns,
        #     advantage,
        #     rm_attention_mask,
        #     rm_action_mask,
        #     info,
        # )
        # critic_experience.to_device("cpu")
        # self._ref = self.critic.append.remote(critic_experience)

        self.actor.train()  # reset model state
        return experience

    # def _generate_local(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
    #     return self.actor.generate(**inputs, **kwargs)
    
    def _generate_local(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        generate_args = {
            "input_ids": inputs['input_ids'],
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": True,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens ", 1),
        }
        if kwargs.get("pad_token", None):
            generate_args["pad_token"] = kwargs.get("pad_token")
        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        outputs = self.actor.model.generate(**generate_args)

        # swap tokens
        rm_outputs = []
        for output in self.tokenizer.batch_decode(outputs):
            if kwargs.get("ultrarm_shift_template"):
                assert "<|user|> " in output and " <|assistant|>\n" in output, f"output: {output}"
                output = output.replace("<|user|> ", "Human: ", 1).replace("<|assistant|>\n", "Assistant: ",1)
            rm_outputs.append(output)
        rm_outputs = self.tokenizer(rm_outputs, return_tensors="pt", padding=True, add_special_tokens=False)['input_ids']

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        return self.actor.process_sequences(outputs, inputs['input_ids'].size(1), eos_token_id, pad_token_id), self.actor.process_sequences(rm_outputs, inputs['input_ids'].size(1), eos_token_id, pad_token_id)


    def _generate_vllm(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm import SamplingParams

        # # testing tokenizer
        # from transformers import LlamaTokenizer
        # tulu_tokenizer = self.tokenizer
        # ultrarm_tokenizer = LlamaTokenizer.from_pretrained("openbmb/UltraRM-13b")
        # for i in range(len(prompts)): assert tulu_tokenizer.encode(prompts[i]) == ultrarm_tokenizer.encode(prompts[i]), "Tokenizer mismatch!"

        pad_token_id, eos_token_id, unk_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, self.tokenizer.unk_token_id

        # round-robin load balance
        rank = torch.distributed.get_rank()
        llm = self.vllm_engines[rank % len(self.vllm_engines)]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 16),
        )

        # TODO: can't pass `max_length` to vLLM's tokenizer for input truncation, remove this once it is supported.
        input_ids = self.tokenize_fn(prompts, self.prompt_max_len, device="cpu")["input_ids"]
        assert self.tokenizer.padding_side == "left", f"tokenizer padding_size should be left"
        pad_indices = (input_ids != pad_token_id).to(dtype=torch.int).argmax(dim=-1)
        prompt_token_ids = []
        for i, pad_index in enumerate(pad_indices.numpy()):
            prompt_token_ids.append(input_ids[i][pad_index:].tolist())
        outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))

        if kwargs.get("ultrarm_shift_template", False):
            for i in range(len(prompts)):
                assert "<|user|>\n" in prompts[i] and "\n<|assistant|>\n" in prompts[i]
                prompts[i] = prompts[i].replace("<|user|>\n", "Human: ", 1).rsplit("\n<|assistant|>\n", 1)[0] + "\nAssistant: "
            rm_input_ids = self.tokenize_fn(prompts, self.prompt_max_len, device="cpu")["input_ids"]
            rm_pad_indices = (rm_input_ids != pad_token_id).to(dtype=torch.int).argmax(dim=-1)
        else:
            # just create new copies
            rm_input_ids = input_ids
            rm_pad_indices = pad_indices
            
        rm_prompt_token_ids = []
        for i, pad_index in enumerate(rm_pad_indices.numpy()):
            rm_prompt_token_ids.append(rm_input_ids[i][pad_index:].tolist())

        def pad_batched_sequences(prompts_tokens, outputs):
            assert len(prompts_tokens) == len(outputs)
            eos_penalty_flags = [False for _ in outputs]
            # NOTE: concat all outputs to following format:
            #
            # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
            # | token token token token token | token token [EOS] [PAD] |
            # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
            # |<---------- prompt ----------->|<-------- answer ------->|
            max_input_len, max_output_len = 0, 0
            for output in outputs:
                # TODO: how to force vLLM generate at least one token?
                output_token_ids = output.outputs[0].token_ids
                if output_token_ids[0] == eos_token_id:
                    logger.warning(f"Only EOS output for prompt: {output.prompt}")
                    output.outputs[0].token_ids = [unk_token_id, eos_token_id]

                max_input_len = max(max_input_len, len(output.prompt_token_ids))
                max_output_len = max(max_output_len, len(output_token_ids))
        
            sequences = []
            for i, (prompt_tokens, output) in enumerate(zip(prompts_tokens, outputs)):
                # left padding input
                input_len = len(prompt_tokens)
                input_ids = [pad_token_id] * (max_input_len - input_len) + prompt_tokens

                # right padding output
                output_len = len(output.outputs[0].token_ids)
                output_ids = output.outputs[0].token_ids + [pad_token_id] * (max_output_len - output_len)
                if output_ids[output_len - 1] != eos_token_id:
                    assert output_len == max_output_len
                    output_ids[-1] = eos_token_id
                    eos_penalty_flags[i] = True # penalize for missing EOS token

                # concat input and output
                sequences.append(input_ids + output_ids)

            sequences = torch.tensor(sequences)
            sequences, attention_mask, action_mask = self.actor.process_sequences(
                sequences, max_input_len, eos_token_id, pad_token_id
            )
            return sequences.to("cuda"), attention_mask.to("cuda"), action_mask.to("cuda"), eos_penalty_flags
        
        return pad_batched_sequences(prompts_tokens=prompt_token_ids, outputs=outputs), pad_batched_sequences(prompts_tokens=rm_prompt_token_ids, outputs=outputs)

    def flush(self):
        "Ensure all experience has been send to critic"
        ray.get(self._ref)
        self._ref = None
