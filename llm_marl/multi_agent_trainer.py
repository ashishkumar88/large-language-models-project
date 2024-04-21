#!/usr/bin/env python

# license TBD

# system imports
import time
import os
from abc import ABC, abstractmethod

# library imports
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# local imports
from llm_marl.logger.logger import create_logger
from llm_marl.utils import Utils

class BaseMultiAgentTrainer(ABC):

    def __init__(self, 
        env=None,
        agent=None,
        base_dir: str = None,
        num_agents: int = 2,
        device = torch.device("cuda:0"),
        experiment_name: str = "ppo",
        episodes_total: int = 10000,
        env_type: str = "gymnasium",
        evaluation_episodes: int = 10,
        evaluation_env=None,
        evaluation_frequency: int = 50,
        ) -> None:
        
        # Training Configs
        self._env = env
        self._agent = agent
        self._base_dir = base_dir
        self._num_agents = num_agents
        self._device = device
        self._experiment_name = experiment_name
        self._logger = create_logger(f"{self._experiment_name}_trainer")
        self._summary_writer = SummaryWriter(os.path.join(base_dir, "tb_logs"))
        self._episodes_total = episodes_total
        self._env_type = env_type
        self._evaluation_episodes = evaluation_episodes
        self._evaluation_env = evaluation_env
        self._evaluation_frequency = evaluation_frequency

        self._total_steps = 0
        self._total_episodes = 0
        self._total_iterations = 0

        # write meta data to base dir
        Utils.write_meta_data(self._base_dir, self)

        # make experiment entry
        Utils.make_experiment_entry(self._base_dir, self._experiment_name)

        # type of gym environment
        self._is_pettingzoo = True if "pettingzoo" in self._env.__module__ else False
        self._is_ma_gym = False
        self._is_lbforaging = False
    
    def __del__(self):
        self._summary_writer.close()

    @abstractmethod
    def train(self):
        raise NotImplementedError("Train method not implemented")   

    @abstractmethod
    def _train(self):
        raise NotImplementedError("Train method not implemented")

    def _evaluate(self, num_episodes=None):
        eval_rewards = []

        if num_episodes is None:
            num_episodes = self._evaluation_episodes

        for _ in range(num_episodes):
            if self._is_pettingzoo:
                episode_reward, _ = self._collect_episode_pettingzoo(eval=True)
            else:
                episode_reward, _ = self._collect_episode_generic(eval=True)

            eval_rewards.append(episode_reward)

        self._logger.info(f"Average evaluation episodic rewards: {sum(eval_rewards)/len(eval_rewards)}")

        return sum(eval_rewards)/len(eval_rewards)
    
    def _do_logging(self, log):
        """ Logs the training progress to the terminal.
        """
        
        for key in log:
            self._logger.info(f"{key}: {log[key]}")

            # tensorboard logging
            self._summary_writer.add_scalar(key, log[key], self._total_iterations)

        self._summary_writer.flush()


    def _collect_episode_pettingzoo(self, env_actions_only=False, eval=False):
        step_count = 0
        episodic_reward = 0

        env = self._evaluation_env if eval else self._env
        observations, _ = env.reset()

        while env.agents:
            if env_actions_only:
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            else:
                obs = np.array([observations[agent] for agent in env.agents])
                observations_dict = Utils.create_observation_dict(obs, self._device)
                actions = self._agent.generate_action(observations_dict, eval)
                actions = {f'agent_{i}': actions[i] for i in range(len(actions))}
            
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            agents = next_observations.keys()

            observations = np.array([observations[agent] for agent in agents])
            next_obs = np.array([next_observations[agent] for agent in agents])
            rewards = np.array([rewards[agent] for agent in agents])
            terminations = np.array([terminations[agent] for agent in agents])
            truncations = np.array([truncations[agent] for agent in agents])
            actions = np.array([actions[agent] for agent in agents])

            # write to replay buffer
            sars = Utils.create_sars(observations, next_obs, actions, rewards, terminations, truncations, self._device)
            self._replay_buffer.extend(sars)
            observations = next_observations

            episodic_reward += sum(rewards)
            step_count += 1

        return episodic_reward, step_count


    def _collect_episode_generic(self, env_actions_only=False, eval=False, max_steps_per_episode=300):
        terminated = False
        step_count = 0
        episodic_reward = 0

        env = self._evaluation_env if eval else self._env
        observations, _ = env.reset()
        
        while not terminated and step_count < max_steps_per_episode:
            if env_actions_only:
                action = Utils.sample_action_from_env(env)
            else:
                obs_dict = Utils.create_observation_dict(np.array(observations), self._device)
                action = self._agent.generate_action(obs_dict, eval)

            next_observations, rewards, terminations, truncations, infos = env.step(action)

            observations = np.array(observations)
            next_obs = np.array(next_observations)
            rewards = np.array(rewards)
            terminations = np.array(terminations)
            truncations = np.array(truncations)
            actions = np.array(action)

            # write to replay buffer
            sars = Utils.create_sars(observations, next_obs, actions, rewards, terminations, truncations, self._device)
            self._replay_buffer.extend(sars)
            observations = next_observations

            episodic_reward += sum(rewards)
            step_count += 1

        return episodic_reward, step_count
