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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from pprint import pprint
from typing import Type, Dict

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import re
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
from search_r1.llm_agent.tool_protocol import QWEN35_NATIVE
from search_r1.trajectory_trace import (TraceJsonlWriter,
                                        parse_search_r1_transcript,
                                        stable_sample_id)

WorkerType = Type[Worker]


def _next_training_step(completed_step, total_steps):
    """Return the next one-based update number, or None after the final update."""
    if completed_step >= total_steps:
        return None
    return completed_step + 1


def _trace_question(decoded_prompt):
    matches = re.findall(
        r'Question:\s*(.*?)(?:<\|im_end\|>|<\|eot_id\|>|$)',
        decoded_prompt,
        flags=re.DOTALL,
    )
    if not matches:
        raise ValueError('trace logging could not extract Question from the prompt')
    question = matches[-1].strip()
    if not question:
        raise ValueError('trace logging extracted an empty question')
    return question


def _json_list(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _get_eval_group_size(config):
    group_size = config.data.get('eval_group_size', 1)
    if (isinstance(group_size, bool)
            or not isinstance(group_size, (int, np.integer))
            or group_size < 1):
        raise ValueError('data.eval_group_size must be a positive integer')
    return int(group_size)


def _prepare_validation_batch(batch, group_size):
    """Repeat validation prompts only for an explicitly grouped evaluation."""
    if group_size == 1:
        return batch
    prompt_batch_size = len(batch)
    repeated = batch.repeat(repeat_times=group_size, interleave=True)
    repeated.non_tensor_batch['group_slot'] = np.tile(
        np.arange(group_size, dtype=object), prompt_batch_size)
    repeated.non_tensor_batch['group_size'] = np.full(
        len(repeated), group_size, dtype=object)
    return repeated


def _validation_meta_info(tokenizer, group_size):
    return {
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'recompute_log_prob': False,
        'do_sample': group_size > 1,
        'validate': True,
    }


def _grouped_eval_record_id(stage, group_uid, group_slot):
    identity = json.dumps(
        {
            'record_type': 'eval',
            'stage': stage,
            'group_uid': group_uid,
            'group_slot': group_slot,
        },
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(identity.encode('ascii')).hexdigest()
    return f'trace:{digest[:24]}'


def _event_aligned_trace_turns(generation_events, retrieval_events,
                               search_count):
    """Build trace turns at generation boundaries using environment events."""
    if len(retrieval_events) != search_count:
        raise ValueError(
            'retrieval event count does not match executed_search_count')

    generation_turn_ids = []
    executed_generation_events = []
    for event in generation_events:
        if not isinstance(event, dict):
            raise ValueError('generation events must be dictionaries')
        turn_id = event.get('turn')
        if isinstance(turn_id, bool) or not isinstance(turn_id, int):
            raise ValueError('generation event turn must be an integer')
        generation_turn_ids.append(turn_id)
        if bool(event.get('executed_search', False)):
            executed_generation_events.append(event)
    if len(set(generation_turn_ids)) != len(generation_turn_ids):
        raise ValueError('generation event turns must be unique')
    if len(executed_generation_events) != search_count:
        raise ValueError(
            'generation event count does not match executed_search_count')

    retrieval_turn_ids = []
    for event in retrieval_events:
        if not isinstance(event, dict):
            raise ValueError('retrieval events must be dictionaries')
        turn_id = event.get('turn')
        if isinstance(turn_id, bool) or not isinstance(turn_id, int):
            raise ValueError('retrieval event turn must be an integer')
        retrieval_turn_ids.append(turn_id)
    if len(set(retrieval_turn_ids)) != len(retrieval_turn_ids):
        raise ValueError('retrieval event turns must be unique')
    if ([event['turn'] for event in executed_generation_events]
            != retrieval_turn_ids):
        raise ValueError('generation and retrieval event turns are not aligned')

    retrieval_by_turn = {
        event['turn']: event for event in retrieval_events
    }
    turns = []
    for event in generation_events:
        event_text = str(event.get('text', ''))
        environment_action = None
        if 'action' in event:
            action = event.get('action')
            content = event.get('content', '')
            parse_error = event.get('parse_error')
            if action not in ('search', 'answer', None):
                raise ValueError('generation event has an unknown action')
            if not isinstance(content, str):
                raise ValueError('generation event content must be a string')
            if parse_error is not None and not isinstance(parse_error, str):
                raise ValueError('generation event parse_error must be a string or None')
            parsed_valid = action in ('search', 'answer') and parse_error is None
            event_turn = {
                'turn': 0,
                'think': str(event.get('reasoning_prefix', '')),
                'action': action if parsed_valid else 'invalid',
                'search_query': content if parsed_valid and action == 'search' else None,
                'answer': content if parsed_valid and action == 'answer' else None,
                'observation': None,
                'invalid_text': [] if parsed_valid else [event_text],
                'valid_action': parsed_valid,
            }
            event_turns = [event_turn]
            if parsed_valid:
                environment_action = event_turn
        else:
            event_turns = parse_search_r1_transcript(event_text)
            if not event_turns:
                event_turns = [{
                    'turn': 0,
                    'think': '',
                    'action': 'invalid',
                    'search_query': None,
                    'answer': None,
                    'observation': None,
                    'invalid_text': [],
                    'valid_action': False,
                }]
            action_match = re.search(r'<(search|answer)>(.*?)</\1>', event_text,
                                     re.DOTALL)
            if action_match is not None:
                action = action_match.group(1)
                content = action_match.group(2).strip()
                value_field = ('search_query'
                               if action == 'search' else 'answer')
                environment_action = next(
                    (turn for turn in event_turns
                     if turn['action'] == action
                     and turn[value_field] == content), None)
                if environment_action is None:
                    environment_action = {
                        'turn': 0,
                        'think': '',
                        'action': action,
                        'search_query': content if action == 'search' else None,
                        'answer': content if action == 'answer' else None,
                        'observation': None,
                        'invalid_text': [],
                        'valid_action': True,
                        'synthetic_environment_action': True,
                    }
                    event_turns.append(environment_action)
            parsed_valid = action_match is not None
        if bool(event.get('valid_action', False)) != parsed_valid:
            raise ValueError(
                'generation event valid_action does not match parsed action')

        for turn in event_turns:
            turn['generation_turn'] = event['turn']
            turn['environment_action'] = turn is environment_action
            turn['synthetic_environment_action'] = bool(
                turn.get('synthetic_environment_action', False))
            turn['retrieved_docs'] = []
            turn['retrieval_executed'] = False

        if bool(event.get('executed_search', False)):
            retrieval_event = retrieval_by_turn[event['turn']]
            if (environment_action is None
                    or environment_action['action'] != 'search'
                    or environment_action['search_query']
                    != retrieval_event.get('query')):
                raise ValueError(
                    'executed generation event does not match retrieval event')
            environment_action['retrieved_docs'] = retrieval_event.get(
                'documents', [])
            environment_action['observation'] = retrieval_event.get(
                'visible_observation', retrieval_event.get('observation'))
            environment_action['retrieval_executed'] = True

        for turn in event_turns:
            turn['turn'] = len(turns)
            turns.append(turn)
    return turns


def _validation_metrics(data_sources, em_scores, search_counts, utilities):
    grouped = defaultdict(lambda: {'em': [], 'searches': [], 'utility': []})
    for data_source, em, searches, utility in zip(
            data_sources, em_scores, search_counts, utilities):
        values = grouped[data_source]
        values['em'].append(float(em))
        values['searches'].append(float(searches))
        values['utility'].append(float(utility))

    metrics = {}
    for data_source, values in grouped.items():
        metrics[f'val/test_score/{data_source}'] = np.mean(values['em'])
        metrics[f'val/utility/{data_source}'] = np.mean(values['utility'])
        metrics[f'val/em/{data_source}'] = np.mean(values['em'])
        metrics[f'val/search_count/{data_source}'] = np.mean(values['searches'])
        metrics[f'val/no_search_ratio/{data_source}'] = np.mean(
            np.asarray(values['searches']) == 0)
    return metrics


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())
    if 'executed_search_count' in batch.batch.keys():
        search_count = batch.batch['executed_search_count'].float()
        metrics['env/executed_search_count/mean'] = search_count.mean().detach().item()
        metrics['env/executed_search_count/max'] = search_count.max().detach().item()
        metrics['env/executed_search_count/min'] = search_count.min().detach().item()
        metrics['env/no_search_ratio'] = (search_count == 0).float().mean().detach().item()
    if 'sequence_em_scores' in batch.batch.keys():
        em_scores = batch.batch['sequence_em_scores'].float()
        metrics['env/em/mean'] = em_scores.mean().detach().item()
    if 'sequence_search_costs' in batch.batch.keys():
        search_costs = batch.batch['sequence_search_costs'].float()
        metrics['env/search_cost/mean'] = search_costs.mean().detach().item()
    if 'sequence_train_rewards' in batch.batch.keys():
        train_rewards = batch.batch['sequence_train_rewards'].float()
        metrics['env/train_reward/mean'] = train_rewards.mean().detach().item()
    if 'sequence_posthoc_utilities' in batch.batch.keys():
        utilities = batch.batch['sequence_posthoc_utilities'].float()
        metrics['env/posthoc_utility/mean'] = utilities.mean().detach().item()
    if all(key in batch.batch.keys() for key in (
            'sequence_em_scores', 'executed_search_count',
            'sequence_train_rewards', 'advantages')) and \
            'uid' in batch.non_tensor_batch:
        metrics.update(_compute_group_metrics(batch))

    return metrics


def _compute_group_metrics(batch):
    em_scores = batch.batch['sequence_em_scores'].float()
    search_counts = batch.batch['executed_search_count'].float()
    rewards = batch.batch['sequence_train_rewards'].float()
    response_width = batch.batch['responses'].shape[-1]
    response_mask = batch.batch['attention_mask'][:, -response_width:].bool()
    group_indices = defaultdict(list)
    for index, uid in enumerate(batch.non_tensor_batch['uid']):
        group_indices[uid].append(index)

    correct_counts = []
    reward_stds = []
    for indices in group_indices.values():
        correct_counts.append(int(em_scores[indices].sum().item()))
        reward_stds.append(float(rewards[indices].std(unbiased=True).item()))
    group_size = max(len(indices) for indices in group_indices.values())
    metrics = {
        'env/group/all_wrong_ratio': float(np.mean(
            np.asarray(correct_counts) == 0)),
        'env/group/reward_std/mean': float(np.mean(reward_stds)),
    }
    for count in range(group_size + 1):
        metrics[f'env/group/correct_count_{count}_ratio'] = float(np.mean(
            np.asarray(correct_counts) == count))

    for label, mask in (('correct', em_scores == 1), ('wrong', em_scores == 0)):
        if mask.any():
            selected_searches = search_counts[mask]
            metrics[f'env/search_count/{label}_mean'] = float(
                selected_searches.mean().item())
            metrics[f'env/no_search_ratio/{label}'] = float(
                (selected_searches == 0).float().mean().item())

    sequence_advantages = []
    for index in range(len(batch)):
        valid = batch.batch['advantages'][index][response_mask[index]]
        if valid.numel():
            sequence_advantages.append(float(valid[0].item()))
    if sequence_advantages:
        metrics.update({
            'env/sequence_advantage/mean': float(np.mean(sequence_advantages)),
            'env/sequence_advantage/min': float(np.min(sequence_advantages)),
            'env/sequence_advantage/max': float(np.max(sequence_advantages)),
        })
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        if config.get('tool_protocol', 'legacy_xml') == QWEN35_NATIVE:
            if not config.get('do_search', False):
                raise ValueError(
                    'qwen35_native requires do_search=true for the agent loop')
            if not config.get('trainer', {}).get('val_only', False):
                raise ValueError(
                    'qwen35_native training is disabled until its '
                    'sampling/log-prob contract is registered')

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()
        self.train_trace_writer = None
        self.eval_trace_writer = None
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _init_trace_writer(self):
        trace_output_dir = self.config.trainer.get('trace_output_dir', None)
        if not trace_output_dir:
            return

        output_dir = Path(str(trace_output_dir))
        stage = str(self.config.trainer.get(
            'trace_stage', self.config.trainer.experiment_name))
        run_id = str(self.config.trainer.get(
            'trace_run_id', self.config.trainer.experiment_name))
        if self.config.trainer.get('val_only', False):
            checkpoint_digest = self.config.trainer.get(
                'trace_checkpoint_digest', None)
            if not checkpoint_digest:
                raise ValueError(
                    'trainer.trace_checkpoint_digest is required for evaluation traces')
            expected_rows = (len(self.val_dataloader) * int(
                self.config.data.val_batch_size) * _get_eval_group_size(
                    self.config))
            self.eval_trace_writer = TraceJsonlWriter(
                output_dir / 'eval_predictions.jsonl',
                record_type='eval',
                expected_rows=expected_rows,
                run_id=run_id,
                stage=stage,
            )
        else:
            expected_rows = (
                int(self.total_training_steps)
                * int(self.config.data.train_batch_size)
                * int(self.config.actor_rollout_ref.rollout.n_agent)
                * int(self.config.actor_rollout_ref.rollout.n)
            )
            self.train_trace_writer = TraceJsonlWriter(
                output_dir / 'train_trajectories.jsonl',
                record_type='train',
                expected_rows=expected_rows,
                run_id=run_id,
                stage=stage,
            )

    def _common_trace_records(self, batch):
        response_width = batch.batch['responses'].shape[-1]
        prompt_width = batch.batch['prompts'].shape[-1]
        records = []
        for index in range(len(batch)):
            item = batch[index]
            prompt_length = int(
                item.batch['attention_mask'][:prompt_width].sum().item())
            response_length = int(
                item.batch['attention_mask'][prompt_width:].sum().item())
            prompt_ids = item.batch['prompts'][-prompt_length:]
            response_ids = item.batch['responses'][:response_length]
            decoded_prompt = self.tokenizer.decode(prompt_ids)
            raw_trajectory = self.tokenizer.decode(
                torch.cat((prompt_ids, response_ids)))

            extra_info = item.non_tensor_batch.get('extra_info', {})
            if not isinstance(extra_info, dict):
                raise ValueError('trace extra_info must be a dictionary')
            source_index = extra_info.get(
                'index', item.non_tensor_batch.get('index'))
            source_split = extra_info.get('split')
            if source_split == 'val':
                source_split = 'train'
            data_source = str(item.non_tensor_batch['data_source'])
            if source_split not in ('train', 'test'):
                raise ValueError(
                    f'trace source split must be train or test, got {source_split!r}')
            reward_model = item.non_tensor_batch['reward_model']
            gold_answers = _json_list(
                reward_model['ground_truth']['target'])
            if not all(isinstance(answer, str) for answer in gold_answers):
                raise ValueError('trace gold answers must be strings')

            retrieval_events = item.non_tensor_batch.get(
                'retrieval_events', [])
            generation_events = item.non_tensor_batch.get(
                'generation_events', [])
            retrieval_events = (
                [] if retrieval_events is None else _json_list(retrieval_events))
            generation_events = (
                [] if generation_events is None else _json_list(generation_events))
            search_count = int(
                item.batch['executed_search_count'].item())
            turns = _event_aligned_trace_turns(generation_events,
                                                retrieval_events, search_count)
            answers = [
                turn['answer'] for turn in turns
                if turn['action'] == 'answer' and turn['answer'] is not None
            ]
            if 'final_answer' in item.non_tensor_batch:
                final_answer = item.non_tensor_batch['final_answer']
                if final_answer is not None and not isinstance(final_answer, str):
                    raise ValueError('trace final_answer must be a string or None')
                event_answer = answers[-1] if answers else None
                if final_answer != event_answer:
                    raise ValueError(
                        'trace final_answer does not match generation events')

            records.append({
                'sample_id': stable_sample_id(
                    data_source, source_split, source_index),
                'source_index': int(source_index),
                'source_split': source_split,
                'data_source': data_source,
                'question': _trace_question(decoded_prompt),
                'gold_answers': gold_answers,
                'raw_trajectory': raw_trajectory,
                'turns': turns,
                'extracted_answer': answers[-1] if answers else None,
                'em': int(item.batch['sequence_em_scores'].item()),
                'executed_search_count': search_count,
                'posthoc_utility': float(
                    item.batch['sequence_posthoc_utilities'].item()),
                'response_tokens': response_length,
                'response_clipped': any(
                    bool(event['clipped']) for event in generation_events),
                'turns_used': len(turns),
                'invalid_action_count': sum(
                    not turn['valid_action'] for turn in turns),
                'retrieval_events': retrieval_events,
                'generation_events': generation_events,
                'generated_tokens': sum(
                    int(event['token_count']) for event in generation_events),
                'generation_clipped_count': sum(
                    bool(event['clipped']) for event in generation_events),
                'environment_invalid_action_count': sum(
                    event.get('valid_action') is False
                    for event in generation_events),
                'unfinished_generation_count': sum(
                    event.get('done') is False
                    for event in generation_events),
                'trajectory_capacity_tokens': int(
                    self.config.data.max_prompt_length),
                'reward_mode': str(self.config.algorithm.get(
                    'cost_reward_mode', 'linear')),
                'cost_lambda': float(self.config.algorithm.get(
                    'cost_lambda', 0.0)),
                'max_searches': int(self.config.max_turns),
                'response_width': int(response_width),
            })
        return records

    def _append_eval_traces(self, batch):
        if self.eval_trace_writer is None:
            return
        checkpoint_digest = str(
            self.config.trainer.trace_checkpoint_digest)
        group_size = _get_eval_group_size(self.config)
        records = self._common_trace_records(batch)
        for index, record in enumerate(records):
            if group_size > 1:
                group_slot = int(
                    batch.non_tensor_batch['group_slot'][index])
                recorded_group_size = int(
                    batch.non_tensor_batch['group_size'][index])
                if recorded_group_size != group_size:
                    raise ValueError(
                        'eval trace group_size does not match configuration')
                group_uid = record['sample_id']
                record.update({
                    'group_uid': group_uid,
                    'group_slot': group_slot,
                    'group_size': group_size,
                    'record_id': _grouped_eval_record_id(
                        self.eval_trace_writer.stage, group_uid, group_slot),
                })
            record['checkpoint_digest'] = checkpoint_digest
            self.eval_trace_writer.append(record)

    def _append_train_traces(self, batch):
        if self.train_trace_writer is None:
            return

        records = self._common_trace_records(batch)
        group_indices = defaultdict(list)
        for index, record in enumerate(records):
            group_indices[record['sample_id']].append(index)
        expected_group_size = int(
            self.config.actor_rollout_ref.rollout.n_agent)
        response_mask = batch.batch['attention_mask'][:, -batch.batch['responses'].shape[-1]:]
        parent_digest = self.config.trainer.get(
            'trace_parent_checkpoint_digest', None)

        for sample_id, indices in group_indices.items():
            if len(indices) != expected_group_size:
                raise ValueError(
                    f'trace group {sample_id} has {len(indices)} trajectories, '
                    f'expected {expected_group_size}')
            rewards = batch.batch['sequence_train_rewards'][indices].float()
            em_scores = batch.batch['sequence_em_scores'][indices].float()
            reward_mean = float(rewards.mean().item())
            reward_std = float(rewards.std(unbiased=True).item())
            correct_count = int(em_scores.sum().item())
            for index in indices:
                valid_advantages = batch.batch['advantages'][index][
                    response_mask[index].bool()]
                if valid_advantages.numel() == 0:
                    raise ValueError('trace trajectory has no valid response tokens')
                record = records[index]
                record.update({
                    'step': int(self.global_steps),
                    'group_uid': f'{self.global_steps}:{sample_id}',
                    'group_slot': int(
                        batch.non_tensor_batch['group_slot'][index]),
                    'reward_em_only': float(
                        batch.batch['sequence_em_scores'][index].item()),
                    'train_reward': float(
                        batch.batch['sequence_train_rewards'][index].item()),
                    'group_correct_count': correct_count,
                    'group_reward_mean': reward_mean,
                    'group_reward_std': reward_std,
                    'sequence_advantage': float(valid_advantages[0].item()),
                })
                if parent_digest:
                    record['parent_checkpoint_digest'] = str(parent_digest)
                self.train_trace_writer.append(record)
        self.train_trace_writer.sync()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         tool_protocol=self.config.get(
                                             'tool_protocol', 'legacy_xml'),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       tool_protocol=self.config.get(
                                           'tool_protocol', 'legacy_xml'),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        em_score_lst = []
        search_count_lst = []
        utility_lst = []
        data_source_lst = []
        eval_group_size = _get_eval_group_size(self.config)

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            tool_protocol=self.config.get('tool_protocol', 'legacy_xml'),
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)
                test_batch = _prepare_validation_batch(
                    test_batch, eval_group_size)
                raw_messages = test_batch.non_tensor_batch.get('raw_prompt')

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = _validation_meta_info(
                    self.tokenizer, eval_group_size)

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)

                sequence_utility = reward_tensor.sum(-1)
                em_score_lst.append(test_batch.batch.get('sequence_em_scores', sequence_utility))
                search_count_lst.append(
                    test_batch.batch.get('executed_search_count', torch.zeros_like(sequence_utility)))
                utility_lst.append(test_batch.batch.get(
                    'sequence_posthoc_utilities', sequence_utility))
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                self._append_eval_traces(test_batch)
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                test_batch = _prepare_validation_batch(
                    test_batch, eval_group_size)
                raw_messages = test_batch.non_tensor_batch.get('raw_prompt')
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = _validation_meta_info(
                    self.tokenizer, eval_group_size)
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                            raw_messages=raw_messages,
                        )
                    
                    test_batch = test_batch.union(final_gen_batch_output)
                    
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch)

                    sequence_utility = reward_tensor.sum(-1)
                    em_score_lst.append(test_batch.batch.get('sequence_em_scores', sequence_utility))
                    search_count_lst.append(
                        test_batch.batch.get('executed_search_count', torch.zeros_like(sequence_utility)))
                    utility_lst.append(test_batch.batch.get(
                        'sequence_posthoc_utilities', sequence_utility))
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                    self._append_eval_traces(test_batch)

        em_scores = torch.cat(em_score_lst, dim=0).float().cpu()
        search_counts = torch.cat(search_count_lst, dim=0).float().cpu()
        utilities = torch.cat(utility_lst, dim=0).float().cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)
        metric_dict = _validation_metrics(
            data_sources, em_scores, search_counts, utilities)

        if self.eval_trace_writer is not None:
            self.eval_trace_writer.finalize()
            self.eval_trace_writer = None

        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0
        self._init_trace_writer()
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        if self.total_training_steps <= 0:
            return

        # Training updates are numbered from one; step N must execute before
        # the total-training-steps stop condition is evaluated.
        self.global_steps = 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            tool_protocol=self.config.get('tool_protocol', 'legacy_xml'),
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        # start training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}
                validated_this_step = False

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                prompt_batch_size = len(batch)
                agent_count = int(self.config.actor_rollout_ref.rollout.n_agent)
                batch = batch.repeat(repeat_times=agent_count, interleave=True)
                batch.non_tensor_batch['group_slot'] = np.tile(
                    np.arange(agent_count, dtype=object), prompt_batch_size)
                raw_messages = batch.non_tensor_batch.get('raw_prompt')

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # original code here

                with _timer('step', timing_raw):
                    if not self.config.do_search:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    else:
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        with _timer('gen', timing_raw):
                            generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                                raw_messages=raw_messages,
                            )

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in final_gen_batch_output.batch.keys():
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                                            
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if key != 'old_log_probs':
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    self._append_train_traces(batch)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)
                        validated_this_step = True

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                next_step = _next_training_step(self.global_steps, self.total_training_steps)
                if next_step is None:

                    # perform validation after training
                    if self.val_reward_fn is not None and not validated_this_step:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.train_trace_writer is not None:
                        self.train_trace_writer.finalize()
                        self.train_trace_writer = None
                    return
                self.global_steps = next_step
    
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics
