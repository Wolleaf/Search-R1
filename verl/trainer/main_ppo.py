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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from search_r1.llm_agent.tool_protocol import (LEGACY_XML, QWEN35_NATIVE,
                                                normalize_tool_protocol)
import re
import numpy as np

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self,
                 tokenizer,
                 num_examine,
                 format_score=0.,
                 cost_lambda=0.,
                 max_searches=1,
                 cost_reward_mode='linear',
                 tool_protocol=LEGACY_XML) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.cost_lambda = float(cost_lambda)
        self.max_searches = int(max_searches)
        self.cost_reward_mode = str(cost_reward_mode)
        self.tool_protocol = normalize_tool_protocol(tool_protocol)
        if not np.isfinite(self.cost_lambda) or self.cost_lambda < 0:
            raise ValueError('cost_lambda must be finite and non-negative')
        if self.max_searches <= 0:
            raise ValueError('max_searches must be positive')
        if self.cost_reward_mode not in ('linear', 'correct_only'):
            raise ValueError(
                "cost_reward_mode must be 'linear' or 'correct_only'")

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        em_scores = torch.zeros(len(data), dtype=torch.float32, device=reward_tensor.device)

        if 'executed_search_count' in data.batch.keys():
            search_counts = data.batch['executed_search_count'].to(device=reward_tensor.device, dtype=torch.float32)
        elif self.cost_lambda == 0:
            search_counts = torch.zeros(len(data), dtype=torch.float32, device=reward_tensor.device)
        else:
            raise KeyError('executed_search_count is required when cost_lambda is non-zero')

        if search_counts.shape != (len(data),):
            raise ValueError(
                f'executed_search_count must have shape ({len(data)},), got {tuple(search_counts.shape)}')
        if torch.any(search_counts < 0):
            raise ValueError('executed_search_count must be non-negative')
        if torch.any(search_counts > self.max_searches):
            raise ValueError('executed_search_count exceeds max_searches')
        search_costs = self.cost_lambda * search_counts / self.max_searches
        train_rewards = torch.zeros_like(em_scores)
        posthoc_utilities = torch.zeros_like(em_scores)

        # all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            if self.tool_protocol == QWEN35_NATIVE:
                if 'final_answer' not in data.non_tensor_batch:
                    raise KeyError(
                        'final_answer is required for qwen35_native reward')
                final_answer = data_item.non_tensor_batch['final_answer']
                if final_answer is not None and not isinstance(final_answer, str):
                    raise ValueError('native final_answer must be a string or None')
                if final_answer is None or not final_answer.strip():
                    score = 0.0
                elif qa_em.em_check(final_answer, ground_truth['target']):
                    score = 1.0
                else:
                    score = self.format_score
            else:
                score = compute_score_fn(
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    format_score=self.format_score,
                )

            em_scores[i] = score
            posthoc_utilities[i] = score - search_costs[i]
            if self.cost_reward_mode == 'correct_only':
                train_rewards[i] = score * (1.0 - search_costs[i])
            else:
                train_rewards[i] = posthoc_utilities[i]
            reward_tensor[i, valid_response_length - 1] = train_rewards[i]
            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        # Sequence-level components make EM and utility observable without
        # decoding the response a second time in the trainer.
        data.batch['sequence_em_scores'] = em_scores
        data.batch['sequence_search_costs'] = search_costs
        data.batch['sequence_train_rewards'] = train_rewards
        data.batch['sequence_posthoc_utilities'] = posthoc_utilities
        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.random_utils import seed_everything

    # This process owns data-loader shuffling and rule-based reward evaluation.
    seed_everything(config.trainer.get('seed', 42))

    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    # Validation only generates responses; avoid loading a duplicate reference
    # policy that is never queried on this path.
    if not config.trainer.get('val_only', False):
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer,
                              num_examine=0,
                              cost_lambda=config.algorithm.get('cost_lambda', 0.0),
                              max_searches=config.max_turns,
                              cost_reward_mode=config.algorithm.get(
                                  'cost_reward_mode', 'linear'),
                              tool_protocol=config.get('tool_protocol',
                                                       LEGACY_XML))

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer,
                                  num_examine=1,
                                  cost_lambda=config.algorithm.get('cost_lambda', 0.0),
                                  max_searches=config.max_turns,
                                  cost_reward_mode='linear',
                                  tool_protocol=config.get('tool_protocol',
                                                           LEGACY_XML))

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
