#!/usr/bin/env python

# license TBD

# system imports
from abc import ABC, abstractmethod
import time
import os

# library imports
import torch
from torch._C import device
from tqdm import tqdm

# local imports
from llm_marl.logger.logger import create_logger
from llm_marl.utils import Utils
from llm_marl.multi_agent_trainer import BaseMultiAgentTrainer

class OffPolicyTrainerIndependentAgents(BaseMultiAgentTrainer):
    
    def __init__(self, 
        env=None,
        agent=None,
        base_dir: str = None,
        num_agents: int = 2,
        device = torch.device("cuda:0"),
        experiment_name: str = "sac",
        episodes_total: int = 100000,
        env_type: str = "gymnasium",
        initialization_steps = None,
        replay_buffer = None,
        number_episodes_per_update: int = 10,
        stop_mean_reward: float = 100.0,
        evaluation_env=None,
        evaluation_episodes: int = 10,
        evaluation_frequency: int = 50,
            ) -> None:
    
        super().__init__(
            env=env,
            agent=agent,
            base_dir=base_dir,
            num_agents=num_agents,
            device=device,
            experiment_name=experiment_name,
            episodes_total=episodes_total,
            env_type=env_type,
            evaluation_env=evaluation_env,
            evaluation_episodes=evaluation_episodes,
            evaluation_frequency=evaluation_frequency,
        )

        self._initialization_steps = initialization_steps
        self._replay_buffer = replay_buffer
        self._number_episodes_per_update = number_episodes_per_update
        self._stop_mean_reward = stop_mean_reward
        self._total_episodes = 0

    def _normalize_observation(self, observation):
        """ Normalize observation
        """
        return observation
        
    def train(self):
        """ Training loop
        """

        observations, _ = Utils.perform_env_reset(self._env)        
        observations = self._normalize_observation(observations)

        done = False
        start_time = time.time()    

        # initialize the replay buffer
        self._collect_trajectories(max_steps=self._initialization_steps, env_actions_only=True)
            
        self._logger.info(f"Initialization took {time.time() - start_time} seconds.")
        start_time = time.time()  
        self._logger.info(f"Starting training for {self._episodes_total} episodes.")  

        # initialize 
        start_time = time.time()   
        self._total_episodes = 0
        self._total_iterations = 0

        # train the agents 
        while self._total_episodes < self._episodes_total:
            
            mean_episodic_reward = self._collect_trajectories()
            log = self._agent.perform_updates(self._replay_buffer, self._total_steps)
            self._total_iterations += 1
            log["mean_training_episodic_reward"]  = mean_episodic_reward

            self._logger.info(f"Iteration took {time.time() - start_time} seconds. Total episodes: {self._total_episodes}. Iteration: {self._total_iterations}.")

            # create a checkpoint
            self._agent.save_checkpoint(os.path.join(self._base_dir, "checkpoints", "models.pth"))

            # perform model evaluation
            if self._evaluation_frequency > 0 and self._total_episodes % self._evaluation_frequency == 0:
                mean_evaluation_reward = self._evaluate()
                log["mean_evaluation_episodic_reward"] = mean_evaluation_reward

            # write to tensorboard and logger
            self._do_logging(log)
            start_time = time.time()

    def _train(self):
        self._logger.warning("Not implemented.")

    def _collect_trajectories(self, max_steps=None, env_actions_only=False):

        episodic_rewards = []

        collector = self._collect_episode_pettingzoo if self._is_pettingzoo else self._collect_episode_generic

        if max_steps is None:
            for _ in tqdm(range(self._number_episodes_per_update), desc="Trajectory collection", unit="episodes"):

                episodic_reward, steps = collector(env_actions_only)
                self._total_episodes += 1
                episodic_rewards.append(episodic_reward)

            return sum(episodic_rewards)/len(episodic_rewards)        
        else:
            total_steps = 0
            while total_steps < max_steps:
                episodic_reward, steps = collector(env_actions_only)
                total_steps += steps

class OffPolicyTrainerCTDEAgents(OffPolicyTrainerIndependentAgents):

    def __init__(self, 
        env=None, 
        agent=None, 
        base_dir: str = None, 
        num_agents: int = 2, 
        device=torch.device("cuda:0"), 
        experiment_name: str = "sac", 
        episodes_total: int = 100000, 
        env_type: str = "gymnasium", 
        initialization_steps: int = 1000, 
        replay_buffer=None, 
        number_episodes_per_update: int = 100, 
        stop_mean_reward: float = 100, 
        evaluation_env=None, 
        evaluation_episodes: int = 10, 
        evaluation_frequency: int = 50) -> None:
        
        super().__init__(env, agent, base_dir, num_agents, device, experiment_name, episodes_total, env_type, initialization_steps, replay_buffer, number_episodes_per_update, stop_mean_reward, evaluation_env, evaluation_episodes, evaluation_frequency)