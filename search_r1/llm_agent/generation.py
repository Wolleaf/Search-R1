import torch
import re
import numpy as np
from copy import deepcopy
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from .tool_protocol import (LEGACY_XML, QWEN35_NATIVE, ParsedAction,
                            Qwen35Conversation, normalize_tool_protocol,
                            parse_action)
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    tool_protocol: str = LEGACY_XML

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.tool_protocol = normalize_tool_protocol(
            getattr(config, 'tool_protocol', LEGACY_XML))
        if self.tool_protocol == QWEN35_NATIVE:
            self._validate_native_right_side_capacity()

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _validate_native_right_side_capacity(self) -> None:
        """Ensure PPO can retain every generated token across all turns."""
        response_length = self.config.max_response_length
        required_length = (
            self.config.max_turns
            * (response_length + self.config.max_obs_length)
            + response_length
        )
        if required_length > self.config.max_prompt_length:
            raise ValueError(
                'qwen35_native right-side capacity requires '
                f'{required_length} tokens, but max_prompt_length is '
                f'{self.config.max_prompt_length}')

    def _batch_tokenize(self, responses: List[str],
                        device=None) -> torch.Tensor:
        """Tokenize a batch of responses."""
        input_ids = self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
        return input_ids.to(device) if device is not None else input_ids

    def _current_tool_protocol(self) -> str:
        config = getattr(self, 'config', None)
        return normalize_tool_protocol(
            getattr(self, 'tool_protocol',
                    getattr(config, 'tool_protocol', LEGACY_XML)))

    def _pad_token_rows(self, rows: List[List[int]], device=None) -> torch.Tensor:
        """Right-pad variable-length token rows without re-tokenizing them."""
        width = max((len(row) for row in rows), default=0)
        output = torch.full(
            (len(rows), width),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        for index, row in enumerate(rows):
            if row:
                output[index, :len(row)] = torch.tensor(
                    row, dtype=torch.long, device=device)
        return output

    def _postprocess_native_responses(
            self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Drop post-EOS padding while preserving every sampled token."""
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        token_rows = []
        response_texts = []
        for response in responses:
            sampled = []
            text_tokens = []
            for token_id in response.tolist():
                token_id = int(token_id)
                if token_id == eos_token_id:
                    sampled.append(token_id)
                    break
                if token_id == pad_token_id:
                    break
                sampled.append(token_id)
                text_tokens.append(token_id)
            token_rows.append(sampled)
            try:
                text = self.tokenizer.decode(
                    text_tokens,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                text = self.tokenizer.decode(
                    text_tokens, skip_special_tokens=False)
            response_texts.append(text)
        return self._pad_token_rows(token_rows, responses.device), response_texts

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        if self._current_tool_protocol() == QWEN35_NATIVE:
            if self.config.no_think_rl:
                raise ValueError('stop')
            return self._postprocess_native_responses(responses)

        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(
            responses_str, device=responses.device)
        return responses, responses_str

    def _prepare_native_conversations(self, gen_batch: DataProto,
                                      raw_messages) -> List[Qwen35Conversation]:
        if raw_messages is None:
            raw_messages = gen_batch.non_tensor_batch.get('raw_prompt')
        if raw_messages is None:
            raise ValueError(
                'qwen35_native generation requires batch-aligned raw messages')
        if isinstance(raw_messages, np.ndarray):
            raw_messages = raw_messages.tolist()
        if len(raw_messages) != len(gen_batch):
            raise ValueError('raw messages are not batch-aligned')

        conversations = []
        input_ids = gen_batch.batch['input_ids']
        attention_mask = gen_batch.batch['attention_mask']
        for index, messages in enumerate(raw_messages):
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            prompt_ids = input_ids[index][attention_mask[index].bool()].tolist()
            if len(prompt_ids) > self.config.max_start_length:
                raise ValueError(
                    'native prompt exceeds max_start_length and would make '
                    'rollout/log-prob contexts disagree')
            conversations.append(
                Qwen35Conversation(
                    self.tokenizer,
                    deepcopy(messages),
                    prompt_ids,
                ))
        return conversations

    @staticmethod
    def _parsed_action_record(turn: int, parsed: ParsedAction) -> Dict[str, Any]:
        return {
            'turn': int(turn),
            'action': parsed.action if parsed.valid else 'invalid',
            'content': parsed.content,
            'parse_error': parsed.error,
            'reasoning_prefix': parsed.prefix,
        }

    def _process_native_followups(
        self,
        conversations: List[Qwen35Conversation],
        response_ids: torch.Tensor,
        responses_str: List[str],
        parsed_actions: List[ParsedAction],
        observations: List[str],
        active_mask: torch.Tensor,
        device=None,
    ) -> Tuple[torch.Tensor, List[str]]:
        suffix_rows = []
        visible_observations = [''] * len(responses_str)
        for index, active in enumerate(active_mask.tolist()):
            parsed = parsed_actions[index]
            if active and (parsed.action == 'search' or not parsed.valid):
                sampled_ids = []
                for token_id in response_ids[index].tolist():
                    token_id = int(token_id)
                    if token_id == self.tokenizer.eos_token_id:
                        sampled_ids.append(token_id)
                        break
                    if token_id == self.tokenizer.pad_token_id:
                        break
                    sampled_ids.append(token_id)
                followup = conversations[index].append_followup(
                    responses_str[index],
                    parsed,
                    observations[index],
                    self.config.max_obs_length,
                    response_token_ids=sampled_ids,
                )
                suffix_rows.append(list(followup.token_ids))
                visible_observations[index] = followup.visible_observation
            else:
                suffix_rows.append([])
        return (self._pad_token_rows(suffix_rows, device=device),
                visible_observations)

    def _process_next_obs(
            self, next_obs: List[str],
            device=None) -> Tuple[torch.Tensor, List[str]]:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        visible_observations = self.tokenizer.batch_decode(
            next_obs_ids, skip_special_tokens=True)
        if len(visible_observations) != len(next_obs):
            raise ValueError('decoded observations are not batch-aligned')
        if device is not None:
            next_obs_ids = next_obs_ids.to(device)
        return next_obs_ids, visible_observations

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Native PPO replays the initial left side plus the complete policy
        # right side, so rollout generation must retain the same context.
        effective_len = new_attention_mask.sum(dim=1).max()
        rolling_capacity = self.config.max_prompt_length
        if self._current_tool_protocol() == QWEN35_NATIVE:
            rolling_capacity += self.config.max_start_length
        max_len = min(rolling_capacity, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        if (self._current_tool_protocol() == QWEN35_NATIVE
                and effective_len > self.config.max_prompt_length):
            raise RuntimeError(
                'qwen35_native policy right side exceeded its validated '
                'max_prompt_length capacity')
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(
            padded_batch,
            meta_info=active_batch.meta_info.copy(),
        )
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor,
                     raw_messages=None) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        native_protocol = self._current_tool_protocol() == QWEN35_NATIVE
        conversations = (self._prepare_native_conversations(
            gen_batch, raw_messages) if native_protocol else None)
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        executed_search_count = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.long)
        retrieval_events = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]
        generation_events = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]
        final_answers = [None for _ in range(gen_batch.batch['input_ids'].shape[0])]
        parsed_action_history = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            turn_active_mask = active_mask.clone()
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict(
                {k: v[active_mask] for k, v in rollings.batch.items()},
                meta_info=rollings.meta_info.copy(),
            )
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            for index, active in enumerate(active_mask.tolist()):
                if active:
                    token_count = int((responses_ids[index] != self.tokenizer.pad_token_id).sum().item())
                    generation_events[index].append({
                        'turn': step,
                        'text': responses_str[index],
                        'token_count': token_count,
                        'clipped': token_count >= getattr(
                            self.config, 'max_response_length', responses_ids.shape[1]),
                    })

            # Execute in environment and process observations
            next_obs, dones, valid_action, executed_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            parsed_actions = getattr(self, '_last_parsed_actions', None)
            for index, active in enumerate(active_mask.tolist()):
                if active:
                    generation_events[index][-1].update({
                        'valid_action': bool(valid_action[index]),
                        'done': bool(dones[index]),
                        'executed_search': bool(executed_search[index]),
                    })
                    if native_protocol:
                        parsed = parsed_actions[index]
                        parsed_record = self._parsed_action_record(step, parsed)
                        parsed_action_history[index].append(parsed_record)
                        generation_events[index][-1].update({
                            'tool_protocol': QWEN35_NATIVE,
                            'action': parsed.action if parsed.valid else None,
                            'content': parsed.content,
                            'parse_error': parsed.error,
                            'reasoning_prefix': parsed.prefix,
                        })
                        if parsed.action == 'answer' and parsed.valid:
                            final_answers[index] = parsed.content
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            executed_search_count += torch.tensor(executed_search, dtype=torch.long)

            if native_protocol:
                next_obs_ids, visible_observations = self._process_native_followups(
                    conversations,
                    responses_ids,
                    responses_str,
                    parsed_actions,
                    next_obs,
                    turn_active_mask,
                    device=responses_ids.device,
                )
            else:
                next_obs_ids, visible_observations = self._process_next_obs(
                    next_obs, device=responses_ids.device)
            for index, event in enumerate(
                    self._last_execution_retrieval_events):
                if event is not None:
                    event['turn'] = step
                    event['visible_observation'] = visible_observations[index]
                    retrieval_events[index].append(event)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict(
                {k: v[active_mask] for k, v in rollings.batch.items()},
                meta_info=rollings.meta_info.copy(),
            )
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            for index, active in enumerate(active_mask.tolist()):
                if active:
                    token_count = int((responses_ids[index] != self.tokenizer.pad_token_id).sum().item())
                    generation_events[index].append({
                        'turn': self.config.max_turns,
                        'text': responses_str[index],
                        'token_count': token_count,
                        'clipped': token_count >= getattr(
                            self.config, 'max_response_length', responses_ids.shape[1]),
                    })

            # # Execute in environment and process observations
            _, dones, valid_action, executed_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )
            parsed_actions = getattr(self, '_last_parsed_actions', None)
            for index, active in enumerate(active_mask.tolist()):
                if active:
                    generation_events[index][-1].update({
                        'valid_action': bool(valid_action[index]),
                        'done': bool(dones[index]),
                        'executed_search': bool(executed_search[index]),
                    })
                    if native_protocol:
                        parsed = parsed_actions[index]
                        parsed_record = self._parsed_action_record(
                            self.config.max_turns, parsed)
                        parsed_action_history[index].append(parsed_record)
                        generation_events[index][-1].update({
                            'tool_protocol': QWEN35_NATIVE,
                            'action': parsed.action if parsed.valid else None,
                            'content': parsed.content,
                            'parse_error': parsed.error,
                            'reasoning_prefix': parsed.prefix,
                        })
                        if parsed.action == 'answer' and parsed.valid:
                            final_answers[index] = parsed.content

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        # Keep the legacy aggregate for existing dashboards. Per-example reward
        # code must use the tensor because meta_info is not batch-reordered.
        meta_info['valid_search_stats'] = executed_search_count.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(
            original_left_side,
            original_right_side,
            meta_info,
            executed_search_count,
            retrieval_events,
            generation_events,
            final_answers=final_answers if native_protocol else None,
            parsed_actions=(parsed_action_history
                            if native_protocol else None),
        )

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            executed_search_count: torch.Tensor,
                            retrieval_events=None,
                            generation_events=None,
                            final_answers=None,
                            parsed_actions=None) -> DataProto:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        final_output['executed_search_count'] = executed_search_count
        
        non_tensors = None
        if retrieval_events is not None:
            aligned_events = np.empty(len(retrieval_events), dtype=object)
            aligned_events[:] = retrieval_events
            non_tensors = {'retrieval_events': aligned_events}
        if generation_events is not None:
            if non_tensors is None:
                non_tensors = {}
            aligned_generations = np.empty(len(generation_events), dtype=object)
            aligned_generations[:] = generation_events
            non_tensors['generation_events'] = aligned_generations
        if final_answers is not None:
            if non_tensors is None:
                non_tensors = {}
            aligned_answers = np.empty(len(final_answers), dtype=object)
            aligned_answers[:] = final_answers
            non_tensors['final_answer'] = aligned_answers
        if parsed_actions is not None:
            if non_tensors is None:
                non_tensors = {}
            aligned_actions = np.empty(len(parsed_actions), dtype=object)
            aligned_actions[:] = parsed_actions
            non_tensors['parsed_actions'] = aligned_actions
        final_output = DataProto.from_dict(final_output, non_tensors=non_tensors)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self,
                            predictions: List[str],
                            pad_token: str,
                            active_mask=None,
                            do_search=True) -> Tuple[List[str], List[int], List[int], List[int]]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            Observations, done flags, valid-action flags, and per-example
            executed-search flags.
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, executed_search = [], [], [], []

        if active_mask is None:
            active_mask = [True] * len(cur_actions)
        active_flags = [bool(active) for active in active_mask]
        search_queries = [
            content for action, content, active in zip(cur_actions, contents, active_flags)
            if do_search and active and action == 'search'
        ]
        search_results = self.batch_search(search_queries) if search_queries else []
        search_metadata = getattr(self, '_last_batch_search_metadata', []) if search_queries else []
        assert len(search_results) == len(search_queries)
        if len(search_metadata) != len(search_queries):
            raise ValueError('retrieval metadata is not aligned with search results')
        retrieval_events = []

        for action, content, active in zip(cur_actions, contents, active_flags):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                executed_search.append(0)
                retrieval_events.append(None)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    executed_search.append(0)
                    retrieval_events.append(None)
                elif action == 'search':
                    search_result = search_results.pop(0) if do_search else ''
                    metadata = search_metadata.pop(0) if do_search else None
                    observation = search_result.strip()
                    if self._current_tool_protocol() == QWEN35_NATIVE:
                        next_obs.append(observation)
                    else:
                        next_obs.append(f'\n\n<information>{observation}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    executed_search.append(int(do_search))
                    retrieval_events.append({
                        'query': content,
                        'documents': metadata if metadata is not None else [],
                        'observation': observation,
                    } if do_search else None)
                else:
                    if self._current_tool_protocol() == QWEN35_NATIVE:
                        # The adapter renders a native user retry message.
                        next_obs.append('')
                    else:
                        next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    executed_search.append(0)
                    retrieval_events.append(None)
            
        assert len(search_results) == 0
        assert len(search_metadata) == 0
        self._last_execution_retrieval_events = retrieval_events
            
        return next_obs, dones, valid_action, executed_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
        parsed_actions = []

        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                if self._current_tool_protocol() == QWEN35_NATIVE:
                    parsed = parse_action(prediction, QWEN35_NATIVE)
                    action = parsed.action if parsed.valid else None
                    content = parsed.content
                else:
                    pattern = r'<(search|answer)>(.*?)</\1>'
                    match = re.search(pattern, prediction, re.DOTALL)
                    if match:
                        content = match.group(2).strip()  # Return only the content inside the tags
                        action = match.group(1)
                        parsed = ParsedAction(action, content)
                    else:
                        content = ''
                        action = None
                        parsed = ParsedAction(None, '', 'missing_legacy_action')
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            parsed_actions.append(parsed)

        self._last_parsed_actions = parsed_actions
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> List[str]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        self._last_batch_search_metadata = results
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        response = requests.post(self.config.search_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
