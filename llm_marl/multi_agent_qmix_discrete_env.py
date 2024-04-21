#!/usr/bin/env python

# license TBD

# system imports
import os
import datetime
import argparse

# external imports
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.tensordict import TensorDict
import numpy as np
from torchrl.data.replay_buffers import ReplayBuffer, LazyMemmapStorage
from pettingzoo.mpe import simple_spread_v3

# local imports
from llm_marl.utils import Utils
from llm_marl.qmix import QMIX
from llm_marl.multi_agent_off_policy_trainer import OffPolicyTrainerCTDEAgents

class MultiAgentQMIXExperiment:

    def __init__(self, config_path=None) -> None:
        
        if config_path is None or os.path.isfile(config_path) is False:
            if os.path.isabs(config_path) is False:
                config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", config_path)
            else:
                config_path = config_path
        
        Utils.load_config(config_path, self) # load configs from yaml file as attributes of this class

        # Environment Configs
        self._env = simple_spread_v3.parallel_env(max_cycles=100, render_mode=self._render_mode if getattr(self, "_render_mode", None) is not None else "none")
        self._eval_env = simple_spread_v3.parallel_env(max_cycles=100, render_mode=self._render_mode if getattr(self, "_render_mode", None) is not None else "none")

        # Log Configs
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        if self._base_dir is None:
            self._base_dir = os.path.join(os.environ["HOME"], "MARL", "MAPF")
        
        self._base_dir = os.path.join(self._base_dir, self._experiment_name, timestamp)
        if not os.path.exists(self._base_dir):
            os.makedirs(self._base_dir)
            os.makedirs(os.path.join(self._base_dir, "checkpoints"))

        # Agent Configs
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._qmix_agent = QMIX(
            env=self._env, 
            gamma=self._gamma,
            critic_lr=float(self._critic_lr),
            number_updates_per_iter=self._number_updates_per_iter,
            tau=self._tau,
            device=self._device,
            num_agents=self._num_agents,
            experiment_name=self._experiment_name,
            hidden_cells=self._hidden_cells,
            has_lstm=self._has_lstm,
            has_attention=self._has_attention,
            ensemble_size=self._ensemble_size if hasattr(self, "_ensemble_size") else 0,
            uncertainty_threshold=self._uncertainty_threshold if hasattr(self, "_uncertainty_threshold") else 1,
            use_vicuna=self._use_vicuna if hasattr(self, "_use_vicuna") else False,
            use_oracle=self._use_oracle if hasattr(self, "_use_oracle") else False,
            fine_tune_vicuna=self._fine_tune_vicuna if hasattr(self, "_fine_tune_vicuna") else False,
            base_dir=self._base_dir,
            fine_tune_samples=self._fine_tune_samples if hasattr(self, "_fine_tune_samples") else 1000,
        )
        
        # replay buffer        
        self._replay_buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=self._replay_buffer_size), batch_size=self._batch_size,)

        # Trainer
        self._qmix_trainer = OffPolicyTrainerCTDEAgents(
            env=self._env,
            agent=self._qmix_agent,
            base_dir=self._base_dir,
            num_agents=self._num_agents,
            device=self._device,
            experiment_name=self._experiment_name,
            episodes_total=self._episodes_total,
            env_type=self._env_type,
            replay_buffer=self._replay_buffer,
            initialization_steps=self._initialization_steps,
            number_episodes_per_update=self._number_episodes_per_update,
            stop_mean_reward=self._stop_mean_reward,
            evaluation_env=self._eval_env,
            evaluation_episodes=self._evaluation_episodes,
            evaluation_frequency=self._evaluation_frequency,
        )
    
    def __del__(self):
        self._env.close()
        self._eval_env.close()

    def start(self) -> None:
        """ Training loop for QMIX.
        """
        self._qmix_trainer.train()
        
if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config", type=str, default=None, help="Path to the config file.")
    args = argument_parser.parse_args()

    trainer = MultiAgentQMIXExperiment(args.config)
    trainer.start()