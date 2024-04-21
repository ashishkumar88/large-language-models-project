#!/usr/bin/env python3

# license TBD

# system imports
from typing import Dict
import os
from tensordict import TensorDict
import random
import math

# library imports
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
from torch import Tensor
from torchrl.data import ReplayBuffer
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from fastchat.model import load_model, get_conversation_template, add_model_args
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import transformers
from transformers import Trainer
from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    make_supervised_data_module,
    LazySupervisedDataset,
)

from fastchat.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

# local imports
from llm_marl.mixer import QMixingNetwork
from llm_marl.utils import Utils
from llm_marl.logger.logger import create_logger
    
class AStarOracle:
    """ A* search oracle for multi-agent path planning.
    """
    def __init__(self, env, logger, num_agents: int = 2):
        self._env = env
        self._num_agents = num_agents
        self._logger = logger

    def _a_star_search(self, grid, start, end):
        """ Returns the path as a list of tuples
        """
        try:
            grid = Grid(matrix=grid.tolist())
            start_node = grid.node(start[0], start[1])
            end_node = grid.node(end[0], end[1])
            finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
            path, runs = finder.find_path(start_node, end_node, grid)
            return [(x, y) for x, y in path]
        except:
            return []

    def ask_astar_for_action(self, observation: Tensor) -> int:

        grid, locations = AStarOracle.convert_observation_to_grid(observation)

        # find best action using A* search for each agent
        actions = []

        for i in range(self._num_agents):
            agent_x = locations[i*2]
            agent_y = locations[i*2 + 1]

            # find the closest landmark
            min_dist = float('inf')
            landmark_x_min = 0
            landmark_y_min = 0

            for j in range(self._num_agents):
                landmark_x = locations[j*2 + 6]
                landmark_y = locations[j*2 + 7]
                dist = np.sqrt((agent_x - landmark_x)**2 + (agent_y - landmark_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    landmark_x_min = landmark_x
                    landmark_y_min = landmark_y
            
            # mark other agents as obstacles
            for j in range(self._num_agents):
                if i != j:
                    grid[locations[j*2], locations[j*2 + 1]] = 0 # 0 is an obstacle

            # find the best action using A* search
            path = self._a_star_search(grid, (agent_x, agent_y), (landmark_x_min, landmark_y_min))

            if len(path) > 1:
                # determine the next action based on the path
                x_diff = path[1][0] - agent_x
                y_diff = path[1][1] - agent_y

                best_action = 0
                if x_diff == 0 and y_diff == -1:
                    best_action = 3 # down
                elif x_diff == 0 and y_diff == 1:
                    best_action = 4 # up
                elif x_diff == -1 and y_diff == 0:
                    best_action = 1 # left
                elif x_diff == 1 and y_diff == 0:
                    best_action = 2 # right
                actions.append(best_action)

                # unmark other agents as obstacles
                for j in range(self._num_agents):
                    if i != j:
                        grid[locations[j*2], locations[j*2 + 1]] = 1 # 1 is an empty cell
            else:
                actions.append(self._env.action_spaces[f'agent_{i}'].sample())

        return actions
    
    @staticmethod
    def convert_observation_to_grid(observation: Tensor, grid_multiplier = 100) -> np.ndarray:

        # select the x and y coordinates of the agents and landmarks
        observation_clone = observation.clone().detach().cpu()[:,:,:14]

        # add the x of the agent to the relative positions of the landmarks
        observation_clone[:,:,4] = observation_clone[:,:,4] + observation_clone[:,:,2]
        observation_clone[:,:,5] = observation_clone[:,:,5] + observation_clone[:,:,3]
        observation_clone[:,:,6] = observation_clone[:,:,6] + observation_clone[:,:,2]
        observation_clone[:,:,7] = observation_clone[:,:,7] + observation_clone[:,:,3]
        observation_clone[:,:,8] = observation_clone[:,:,8] + observation_clone[:,:,2]
        observation_clone[:,:,9] = observation_clone[:,:,9] + observation_clone[:,:,3]
        observation_clone[:,:,10] = observation_clone[:,:,10] + observation_clone[:,:,2]
        observation_clone[:,:,11] = observation_clone[:,:,11] + observation_clone[:,:,3]
        observation_clone[:,:,12] = observation_clone[:,:,12] + observation_clone[:,:,2]
        observation_clone[:,:,13] = observation_clone[:,:,13] + observation_clone[:,:,3]

        # find the min and max of the x and y coordinates
        x_col_ids = [2, 4, 6, 8, 10, 12]
        y_col_ids = [3, 5, 7, 9, 11, 13]
        x_min = observation_clone[:,:,x_col_ids].min(dim=2).values.squeeze(0)
        x_min = x_min[0].item()
        x_max = observation_clone[:,:,x_col_ids].max(dim=2).values.squeeze(0)
        x_max = x_max[0].item()
        y_min = observation_clone[:,:,y_col_ids].min(dim=2).values.squeeze(0)
        y_min = y_min[0].item()
        y_max = observation_clone[:,:,y_col_ids].max(dim=2).values.squeeze(0)
        y_max = y_max[0].item()
        
        # calculate the grid size
        x_diff = x_max - x_min
        y_diff = y_max - y_min

        grid = np.ones((int(abs(x_diff)*grid_multiplier) + 1, int(abs(y_diff)*grid_multiplier) + 1))

        # calculate the absolute positions of the agents and landmarks
        agent_1_x = observation_clone[0,0,2].item() - x_min
        agent_1_x = int(abs(agent_1_x)*grid_multiplier)
        agent_1_y = observation_clone[0,0,3].item() - y_min
        agent_1_y = int(abs(agent_1_y)*grid_multiplier)

        agent_2_x = observation_clone[0,1,2].item() - x_min
        agent_2_x = int(abs(agent_2_x)*grid_multiplier)
        agent_2_y = observation_clone[0,1,3].item() - y_min
        agent_2_y = int(abs(agent_2_y)*grid_multiplier)

        agent_3_x = observation_clone[0,2,2].item() - x_min
        agent_3_x = int(abs(agent_3_x)*grid_multiplier)
        agent_3_y = observation_clone[0,2,3].item() - y_min
        agent_3_y = int(abs(agent_3_y)*grid_multiplier)

        landmark_1_x = observation_clone[0,0,4].item() - x_min
        landmark_1_x = int(abs(landmark_1_x)*grid_multiplier)
        landmark_1_y = observation_clone[0,0,5].item() - y_min
        landmark_1_y = int(abs(landmark_1_y)*grid_multiplier)

        landmark_2_x = observation_clone[0,0,6].item() - x_min
        landmark_2_x = int(abs(landmark_2_x)*grid_multiplier)
        landmark_2_y = observation_clone[0,0,7].item() - y_min
        landmark_2_y = int(abs(landmark_2_y)*grid_multiplier)

        landmark_3_x = observation_clone[0,0,8].item() - x_min
        landmark_3_x = int(abs(landmark_3_x)*grid_multiplier)
        landmark_3_y = observation_clone[0,0,9].item() - y_min
        landmark_3_y = int(abs(landmark_3_y)*grid_multiplier)

        locations = [agent_1_x, agent_1_y, agent_2_x, agent_2_y, agent_3_x, agent_3_y, landmark_1_x, landmark_1_y, landmark_2_x, landmark_2_y, landmark_3_x, landmark_3_y]

        # get the bounds of the environment
        return grid, locations

class QMIX:
    """ MultiAgent QMIX algorithm implementation.
    """

    def __init__(self,
            env,
            gamma: float = 0.99,
            critic_lr: float = 3e-4,
            number_updates_per_iter: int = 1,
            device = torch.device("cuda:0"),
            num_agents: int = 2,
            tau: float = 1e-2,
            experiment_name: str = "multi_agent_iql",   
            hidden_cells: list = [256, 256],
            activation_function: nn.Module = nn.ReLU(),
            has_lstm: bool = False,
            has_attention: bool = False,
            mixer_hidden_size: int = 64,
            ensemble_size: int = 0,
            uncertainty_threshold: float = 1,
            use_vicuna: bool = False,
            use_oracle: bool = False,
            fine_tune_vicuna: bool = False,
            fine_tune_samples: int = 1000,
            base_dir: str = None,
    ):

        self._env = env
        self._gamma = gamma
        self._critic_lr = critic_lr
        self._num_agents = num_agents
        self._number_updates_per_iter = number_updates_per_iter
        self._eps_start = 0.95
        self._eps_end = 0.05
        self._eps_decay = 10000
        self._eps_threshold = 1.0
        self._total_steps = 0
        self._tau = tau
        self._device = device
        self._logger = create_logger(f"{experiment_name}_qmix")
        self._base_dir = base_dir

        if not os.path.exists(self._base_dir) or not os.path.isdir(self._base_dir) or self._base_dir is None:
            raise FileNotFoundError("Base directory not found or invalid")

        # qmix modules        
        n_obs, n_act = Utils.get_env_nobs_nact(env)
        self._n_act = n_act

        mixing_net_kwargs = {
            "input_dims": n_obs,
            "out_dims": n_act,
            "hidden_cells": hidden_cells,
            "device": self._device,
            "mixer_hidden_size" : mixer_hidden_size,
            "activation_function": activation_function,
            "num_agents": num_agents,
            "has_lstm": has_lstm,
            "has_attention": has_attention,
        }


        self._q_mixer = QMixingNetwork(**mixing_net_kwargs)
        self._q_mixer.train()
        self._target_q_mixer = QMixingNetwork(**mixing_net_kwargs)
        self._target_q_mixer.eval()

        self._q_mixer_optimizer = Adam(
                self._q_mixer.parameters(),
                lr=self._critic_lr,
            )
        
        # check whether to use vicuna or oracle
        self._use_vicuna = use_vicuna
        self._use_oracle = use_oracle
        
        self._use_ensemble = self._use_vicuna != self._use_oracle and ensemble_size > 1
        if self._use_ensemble:
            self._q_mixer_ensemble = [QMixingNetwork(**mixing_net_kwargs) for _ in range(ensemble_size)]
            self._q_mixer_ensemble_optimizers = [Adam(self._q_mixer_ensemble[i].parameters(), lr=self._critic_lr) for i in range(ensemble_size)]
            self._uncertainty_threshold = uncertainty_threshold

        if self._use_vicuna and self._use_oracle:
            raise ValueError("Cannot use both vicuna and oracle")
        else:
            if self._use_vicuna != self._use_oracle and not self._use_ensemble:
                raise ValueError("Ensemble size must be greater than 1")
        
        if self._use_vicuna:
            self._vicuna_args = dict(model_path='lmsys/vicuna-7b-v1.5', revision='main', device='cuda', gpus=None, num_gpus=1, max_gpu_memory=None, dtype=None, load_8bit=True, cpu_offloading=False, gptq_ckpt=None, gptq_wbits=16, gptq_groupsize=-1, gptq_act_order=False, awq_ckpt=None, awq_wbits=16, awq_groupsize=-1, enable_exllama=False, exllama_max_seq_len=4096, exllama_gpu_split=None, exllama_cache_8bit=False, enable_xft=False, xft_max_seq_len=4096, xft_dtype=None, temperature=0.7, repetition_penalty=1.0, max_new_tokens=1024, debug=False)
        
            self._vicuna_model, self._vicuna_tokenizer = load_model(
                self._vicuna_args["model_path"],
                device=self._vicuna_args["device"],
                num_gpus=self._vicuna_args["num_gpus"],
                max_gpu_memory=self._vicuna_args["max_gpu_memory"],
                load_8bit=self._vicuna_args["load_8bit"],
                cpu_offloading=self._vicuna_args["cpu_offloading"],
                revision=self._vicuna_args["revision"],
                debug=self._vicuna_args["debug"],
            )

            if fine_tune_vicuna:
                self._a_star_oracle = AStarOracle(env, self._logger, num_agents)
                self._fine_tune_vicuna(fine_tune_samples)


        elif self._use_oracle:
            self._a_star_oracle = AStarOracle(env, self._logger, num_agents)

    def _fine_tune_vicuna(self, fine_tune_samples=1000):
        self._logger.info("Fine-tuning vicuna model.")

        # prepare data set
        step_count = 0
        train_dataset = []
        evaluation_dataset = []
        while True:
            observations, _ = self._env.reset()
            while self._env.agents:
                obs = np.array([observations[agent] for agent in self._env.agents])
                observation = Utils.create_observation_dict(obs, self._device)
                actions = self._a_star_oracle.ask_astar_for_action(observation["observation"])
                actions = {f'agent_{i}': actions[i] for i in range(len(actions))}
                next_observations, _, _, _, _ = self._env.step(actions)

                observations = next_observations
                step_count += 1

                conversation = {}
                conversation["conversations"] = []
                conversation["conversations"].append(
                    {
                        "from": "human",
                        "value": self._convert_observation_to_prompt(observation["observation"]),
                    }
                )

                conversation["conversations"].append(
                    {
                        "from": "gpt",
                        "value": str(actions),
                    }   
                )

                train_dataset.append(conversation)

            if step_count >= fine_tune_samples:
                break   

        while True:
            observations, _ = self._env.reset()
            while self._env.agents:
                obs = np.array([observations[agent] for agent in self._env.agents])
                observation = Utils.create_observation_dict(obs, self._device)
                actions = self._a_star_oracle.ask_astar_for_action(observation["observation"])
                actions = {f'agent_{i}': actions[i] for i in range(len(actions))}
                next_observations, _, _, _, _ = self._env.step(actions)

                observations = next_observations
                step_count += 1

                conversation = {}
                conversation["conversations"] = []
                conversation["conversations"].append(
                    {
                        "from": "human",
                        "value": self._convert_observation_to_prompt(observation["observation"]),
                    }
                )

                conversation["conversations"].append(
                    {
                        "from": "gpt",
                        "value": str(actions),
                    }   
                )

                evaluation_dataset.append(conversation)

            if step_count >= fine_tune_samples/10:
                break   

        self._logger.info("Raw data set collected. Initiating fine-tuning.")

        # the fine tuning is inspired by Lora fine tuning in the original Vicuna implementation
        # https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_mem.py

        replace_llama_attn_with_flash_attn()

        training_args = TrainingArguments(
            output_dir=os.path.join(self._base_dir, "vicuna_fine_tuning"),
            fp16=True, 
            learning_rate=2e-5,
            weight_decay=0.01,
            num_train_epochs=10,
            logging_dir=os.path.join(self._base_dir, "vicuna_fine_tuning", "logs"),
            logging_steps=1,
            max_steps=10, # 1 epoch does not help
            report_to="none",)

        config = transformers.AutoConfig.from_pretrained(
            self._vicuna_args["model_path"],
            cache_dir=training_args.cache_dir,
            trust_remote_code=False,
        )

        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        config.use_cache = False

        if self._vicuna_tokenizer.pad_token != self._vicuna_tokenizer.unk_token:
            self._vicuna_tokenizer.pad_token = self._vicuna_tokenizer.unk_token

        # initialize LazySupervisedDataset
        laze_supervised_train_dataset = LazySupervisedDataset(train_dataset, self._vicuna_tokenizer)
        laze_supervised_evaluation_dataset = LazySupervisedDataset(evaluation_dataset, self._vicuna_tokenizer)

        laze_supervised_dataset = dict(train_dataset=laze_supervised_train_dataset, eval_dataset=laze_supervised_evaluation_dataset)

        trainer = Trainer(
            model=self._vicuna_model, tokenizer=self._vicuna_tokenizer, args=training_args, **laze_supervised_dataset
        )
        
        trainer.train()
    
    def _convert_observation_to_prompt(self, observation: Tensor) -> str:
        
        _, locations = AStarOracle.convert_observation_to_grid(observation)

        prompt = f"There are {self._num_agents} agents in the environment. The agents are working in a grid world and all agents are globally rewarded based on how far the closest agent is to each landmark. " \
                   f"Locally, the agents are penalized if they collide with other agents. The possible actions are: 0: nothing, 1: left, 2: right, 3: down, and 4: up. " \
                   f"Please help the agents to plan the next actions given agents' current observations. The actions should be displayed in a list. Do not explain the reasoning. " \
                   f"The first agent is at position {[round(x, 2) for x in locations[0:2]]}, the closest landmarks are at {[round(x, 2) for x in locations[6:]]}. " \
                   f"The second agent is at position {[round(x, 2) for x in locations[2:4]]}, the closest landmarks are at {[round(x, 2) for x in locations[6:]]}. " \
                   f"The third agent is at position {[round(x, 2) for x in locations[4:6]]}, the closest landmarks are at {[round(x, 2) for x in locations[6:]]}. " \
                   f"What are the next actions for the agents? The output should be a list of integers with length {self._num_agents}."
        
        return prompt
    
    def _convert_vicuna_output_to_action(self, outputs: str) -> int:
        actions = Utils.sample_action_from_env(self._env)
        try:
            tmp = eval(outputs)
            if len(tmp) >= self._num_agents and all(isinstance(x, int) for x in tmp):
                actions = tmp[:self._num_agents]
        except:
            pass

        actions = np.array(actions)

        action_min = 0
        action_max = self._n_act - 1
        actions = np.clip(actions, action_min, action_max) 
        return actions

    def _ask_vicuna_for_action(self, observation: Tensor) -> int:
        msg = self._convert_observation_to_prompt(observation)
        conv = get_conversation_template(self._vicuna_args["model_path"])
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        inputs = self._vicuna_tokenizer([prompt], return_tensors="pt").to(self._device)
        output_ids = self._vicuna_model.generate(
            **inputs,
            do_sample=True if self._vicuna_args["temperature"] > 1e-5 else False,
            temperature=self._vicuna_args["temperature"],
            repetition_penalty=self._vicuna_args["repetition_penalty"],
            max_new_tokens=self._vicuna_args["max_new_tokens"],
        )

        if self._vicuna_model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = self._vicuna_tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        return self._convert_vicuna_output_to_action(outputs)

    def get_config(self) -> str:
        config = ""
        config += "gamma: " + str(self._gamma) + "\n"
        config += "critic_lr: " + str(self._critic_lr) + "\n"
        config += "number_updates_per_iter: " + str(self._number_updates_per_iter) + "\n"
        config += "device: " + str(self._device) + "\n"
        config += "num_agents: " + str(self._num_agents) + "\n"
        config += "tau: " + str(self._tau) + "\n"
        config += "Q Mixer network: " + str(self._q_mixer) + "\n"
        return config
            
    def generate_action(self, observation: Tensor, eval=False) -> Dict:
        agent_ids = torch.eye(self._num_agents).unsqueeze(0).expand(observation["observation"].shape[0], -1, -1).to(self._device)
        actions = None
        observation = observation["observation"]
        
        if eval:
            argmax_actions = self._q_mixer.get_argmax_action(observation, agent_ids)
            return argmax_actions.squeeze(0).cpu().numpy() 
        else:
            self._total_steps += 1
            self._eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * \
                np.exp(-1. * self._total_steps / self._eps_decay)
            
            if random.random() > self._eps_threshold:

                # check the uncertainty of the ensemble
                if self._use_ensemble:
                    q_values = torch.stack([model(observation, agent_ids) for model in self._q_mixer_ensemble] + [self._q_mixer(observation, agent_ids)])
                    q_values_mean = q_values.mean(dim=0)
                    q_values_std = q_values.std(dim=0)

                    if q_values_std.mean() > self._uncertainty_threshold:

                        if self._use_vicuna:
                            actions = self._ask_vicuna_for_action(observation)
                        else:
                            actions = self._a_star_oracle.ask_astar_for_action(observation)

                        return actions
                    else:
                        argmax_actions = self._q_mixer.get_argmax_action(observation, agent_ids)
                        return argmax_actions.squeeze(0).cpu().numpy() 
                else:
                    argmax_actions = self._q_mixer.get_argmax_action(observation, agent_ids)
                    return argmax_actions.squeeze(0).cpu().numpy() 
            else:
                actions = Utils.sample_action_from_env(self._env)
                return actions

    def perform_updates(self, replay_buffer: ReplayBuffer, total_steps:int):
        # update the actor and critic networks
        loss_critic = []

        for _ in range(self._number_updates_per_iter):
            with torch.no_grad():
                episode_sars = replay_buffer.sample().to(self._device)
                obs = episode_sars["observation"]
                next_obs = episode_sars[("next", "observation")]
                rewards = episode_sars["reward"].sum(dim=1)
                terminations = torch.all(episode_sars["termination"], dim=1).int()
                agent_ids = torch.eye(self._num_agents).unsqueeze(0).expand(episode_sars.shape[0], -1, -1).to(self._device)
            
                next_q_total = self._target_q_mixer(next_obs, agent_ids)
                y_tot = rewards + (1 - terminations) * self._gamma * next_q_total

            q_total = self._q_mixer(obs, agent_ids)
            loss = F.mse_loss(q_total, y_tot) 
            
            # update the critic network
            self._q_mixer_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self._q_mixer.parameters(), 1)
            self._q_mixer_optimizer.step()

            loss_critic.append(loss.mean().item())

            # update the ensemble
            if self._use_ensemble:
                for i in range(len(self._q_mixer_ensemble)):
                    self._q_mixer_ensemble_optimizers[i].zero_grad()
                    ensemble_loss = F.mse_loss(self._q_mixer_ensemble[i](obs, agent_ids), y_tot)
                    ensemble_loss.backward()
                    clip_grad_norm_(self._q_mixer_ensemble[i].parameters(), 1)
                    self._q_mixer_ensemble_optimizers[i].step()
            
        # write tensorboard logs
        log = {
                'loss_critic' : sum(loss_critic) / len(loss_critic),
            }
        
        # update target network
        for target_param, param in zip(self._target_q_mixer.parameters(), self._q_mixer.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        return log

    def save_checkpoint(self, path: str):
        torch.save(self._q_mixer.state_dict(), path)

    def load_checkpoint(self, path: str):
        if os.path.exists(path) and os.path.isfile(path) and path.endswith(".pth"):
            self._q_mixer.load_state_dict(torch.load(path))
        else:
            raise FileNotFoundError("Checkpoint file not found")
