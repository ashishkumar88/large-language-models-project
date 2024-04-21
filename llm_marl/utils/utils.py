#!/usr/bin/env python

# license TBD

# system imports
import os
import datetime

# library imports
import torch
from tensordict.tensordict import TensorDict
import numpy as np

# local imports
from llm_marl.config.config_loader import load_config

class Utils:

    @staticmethod
    def load_config(config_path, obj):
        trainer_config = load_config(config_path)

        # PPO Trainer Configs
        for config in trainer_config:
            setattr(obj, "_" + config, trainer_config[config])   

    @staticmethod
    def create_sars(observation, next_observation, action, reward, termination, truncation, device):
        sars_dict = {}

        sars_dict["observation"] = torch.from_numpy(np.expand_dims(observation, axis=0)).float().to(device)
        sars_dict["action"] = torch.tensor(action).unsqueeze(-1).to(torch.int64).unsqueeze(0).to(device)
        sars_dict["reward"] = torch.tensor(reward).unsqueeze(-1).to(torch.float32).unsqueeze(0).to(device)
        sars_dict["truncation"] = torch.tensor(truncation).unsqueeze(-1).to(torch.bool).unsqueeze(0).to(device)
        sars_dict["termination"] = torch.tensor(termination).unsqueeze(-1).to(torch.bool).unsqueeze(0).to(device)
        sars_dict[("next", "observation")] = torch.from_numpy(np.expand_dims(next_observation, axis=0)).float().to(device)
        
        sars_dict = TensorDict(sars_dict, 1, device=device)
        return sars_dict
    
    @staticmethod
    def create_observation_dict(observation, device):
        observation_dict = {}
        
        observation_dict["observation"] = torch.from_numpy(np.expand_dims(observation, axis=0)).float().to(device)
        
        observation_dict = TensorDict(observation_dict, 1, device=device)
        return observation_dict
    
    @staticmethod
    def write_meta_data(base_dir, obj):
        with open(base_dir + "/meta_data.txt", "w") as f:
            if hasattr(obj, "_env"):
                f.write("Environment: " + str(obj._env) + "\n")

            if hasattr(obj, "_agent"):
                f.write("Agent: " + str(obj._agent) + "\n")
                f.write("Agent Configs: " + str(obj._agent.get_config()) + "\n")
            
            if hasattr(obj, "_experiment_name"):
                f.write("Experiment Name: " + str(obj._experiment_name) + "\n")

            if hasattr(obj, "_total_steps"):
                f.write("Total Steps: " + str(obj._total_steps) + "\n")
            
            if hasattr(obj, "_total_episodes"):
                f.write("Total Episodes: " + str(obj._total_episodes) + "\n")   
            
    @staticmethod
    def make_experiment_entry(base_dir, experiment_name):
        with open(os.path.join(base_dir, "..", "experiments.txt"), "a") as f:
            f.write(experiment_name + "\t" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "\t" + base_dir + "\n")


    @staticmethod
    def get_env_nobs_nact(env):
        n_obs = None
        n_obs_found = False

        try:
            n_obs = env.observation_spaces['agent_0'].shape[0]
            n_obs_found = True
        except:
            pass

        if not n_obs_found:
            try:
                n_obs = env.observation_space.sample().shape[0]
                n_obs_found = True
            except:
                pass

        if not n_obs_found:
            try:
                n_obs = env.observation_space.sample()[0].shape[0]
                n_obs_found = True
            except:
                pass

        n_act = None
        n_act_found = False

        try:
            n_act = env.action_spaces['agent_0'].n
            n_act_found = True
        except:
            pass

        if not n_act_found:
            try:
                n_act = env.action_space.n
                n_act_found = True
            except:
                pass
        
        if not n_act_found:
            try:
                n_act = env.action_space[0].n
                n_act_found = True
            except:
                pass

        return n_obs, n_act

    @staticmethod
    def sample_action_from_env(env):
        actions = None
        actions_generated = False

        try:
            actions = env.action_space.sample()
            actions_generated = True
        except:
            pass

        if not actions_generated:
            try:
                actions = np.array([env.action_space(agent_id).sample() for agent_id in env.agents])
                actions_generated = True
            except:
                pass
        return actions

    @staticmethod
    def perform_action_in_env(env, actions):
        
        if "pettingzoo" in env.__module__:
            actions = {f'agent_{i}': actions[i] for i in range(len(actions))}

        observations, rewards, terminations, truncations, _ = env.step(actions)
        
        if isinstance(observations, dict):
            observations = np.array([observations[k] for k in env.agents])
            rewards = np.array([rewards[k] for k in env.agents])
            terminations = np.array([terminations[k] for k in env.agents])
            truncations = np.array([truncations[k] for k in env.agents])

        return observations, rewards, terminations, truncations, None

    def perform_env_reset(env):
        observations, info = env.reset()

        if isinstance(observations, dict):
            observations = np.array([observations[k] for k in env.agents])

        return observations, info