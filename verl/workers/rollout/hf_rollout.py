# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
import contextlib
import math
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask
from .base import BaseRollout

from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

__all__ = ['HFRollout']


class _PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Apply OpenAI/vLLM presence semantics to this generation only."""

    def __init__(self, penalty: float, prompt_length: int):
        if (isinstance(penalty, bool)
                or not isinstance(penalty, (int, float))
                or not math.isfinite(float(penalty))
                or not -2.0 <= float(penalty) <= 2.0):
            raise ValueError('presence_penalty must be a finite number in [-2, 2]')
        if (isinstance(prompt_length, bool)
                or not isinstance(prompt_length, int)
                or prompt_length < 0):
            raise ValueError('prompt_length must be a non-negative integer')
        self.penalty = float(penalty)
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.ndim != 2 or scores.ndim != 2:
            raise ValueError('presence penalty expects two-dimensional tensors')
        if input_ids.size(0) != scores.size(0):
            raise ValueError('input_ids and scores must have the same batch size')
        if input_ids.size(1) < self.prompt_length:
            raise ValueError('input_ids are shorter than prompt_length')

        generated_ids = input_ids[:, self.prompt_length:]
        if self.penalty == 0.0 or generated_ids.numel() == 0:
            return scores

        # Scalar scatter gives every seen token one penalty, regardless of count.
        penalties = torch.zeros_like(scores)
        penalties.scatter_(1, generated_ids, self.penalty)
        return scores - penalties


class HFRollout(BaseRollout):

    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get('micro_batch_size', batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']  # left-padded attention_mask
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        pad_token_id = prompts.meta_info['pad_token_id']

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        # make sampling args can be overriden by inputs
        do_sample = prompts.meta_info.get('do_sample', self.config.do_sample)
        response_length = prompts.meta_info.get('response_length', self.config.response_length)
        top_p = prompts.meta_info.get('top_p', self.config.get('top_p', 1.0))
        top_k = prompts.meta_info.get('top_k', self.config.get('top_k', 0))
        min_p = prompts.meta_info.get('min_p', self.config.get('min_p', 0.0))
        presence_penalty = prompts.meta_info.get(
            'presence_penalty', self.config.get('presence_penalty', 0.0))
        repetition_penalty = prompts.meta_info.get(
            'repetition_penalty', self.config.get('repetition_penalty', 1.0))

        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)  # to be compatible with vllm
        if min_p is None:
            min_p = 0.0
        if (isinstance(min_p, bool) or not isinstance(min_p, (int, float))
                or not math.isfinite(float(min_p))
                or not 0.0 <= float(min_p) <= 1.0):
            raise ValueError('min_p must be a finite number in [0, 1]')
        min_p = float(min_p)
        if presence_penalty is None:
            presence_penalty = 0.0
        if (isinstance(presence_penalty, bool)
                or not isinstance(presence_penalty, (int, float))
                or not math.isfinite(float(presence_penalty))
                or not -2.0 <= float(presence_penalty) <= 2.0):
            raise ValueError(
                'presence_penalty must be a finite number in [-2, 2]')
        presence_penalty = float(presence_penalty)
        if repetition_penalty is None:
            repetition_penalty = 1.0
        if (isinstance(repetition_penalty, bool)
                or not isinstance(repetition_penalty, (int, float))
                or not math.isfinite(float(repetition_penalty))
                or float(repetition_penalty) <= 0.0):
            raise ValueError('repetition_penalty must be a finite number greater than 0')
        repetition_penalty = float(repetition_penalty)

        temperature = prompts.meta_info.get('temperature', self.config.temperature)

        generation_config_kwargs = {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'repetition_penalty': repetition_penalty,
        }
        # Transformers runs a full-vocabulary softmax even when min_p is zero.
        if min_p > 0.0:
            generation_config_kwargs['min_p'] = min_p
        generation_config = GenerationConfig(**generation_config_kwargs)

        logits_processor = None
        if presence_penalty != 0.0:
            presence_processor = _PresencePenaltyLogitsProcessor(
                penalty=presence_penalty,
                prompt_length=prompt_length,
            )
            logits_processor = LogitsProcessorList([
                presence_processor,
            ])

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                generate_kwargs = {}
                if logits_processor is not None:
                    generate_kwargs['logits_processor'] = logits_processor
                output = self.module.generate(
                    input_ids=idx,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    # max_length=max_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    # renormalize_logits=True,
                    output_scores=False,  # this is potentially very large
                    return_dict_in_generate=True,
                    use_cache=True,
                    **generate_kwargs)
        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)

        assert seq.shape[1] == sequence_length

        prompt = seq[:, :prompt_length]  # (bs, prompt_length)
        response = seq[:, prompt_length:]  # (bs, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                'prompts': prompt,
                'responses': response,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()

        self.module.train()
        return DataProto(batch=batch)
