#!/usr/bin/env python3

# license TBD

# system imports

# library imports
import torch
from torch import nn
import torch.nn.functional as F

# local imports
from llm_marl.ma_models import DiscreteLinearMultiAgentCritic

class QMixingNetwork(nn.Module):

    def __init__(self, 
        input_dims: int = 1,
        out_dims: int = 1,
        hidden_cells: list = [256, 256],
        mixer_hidden_size: int = 64,
        num_agents: int = 2,
        device = torch.device("cuda:0"),
        activation_function = nn.ReLU(),
        model_has_dropout: bool = False,
        dropout_rate: float = 0.1,
        has_lstm: bool = False,
        has_attention: bool = False,
        ) -> None:

        super(QMixingNetwork, self).__init__()
        
        self._q_network = DiscreteLinearMultiAgentCritic(
            input_dims=input_dims,
            out_dims=out_dims,
            hidden_cells=hidden_cells,
            num_agents=num_agents,
            activation_function=activation_function,
            model_has_dropout=model_has_dropout,
            dropout_rate=dropout_rate,
            has_lstm=has_lstm,
            has_attention=has_attention,
        ).to(device)

        self._num_agents = num_agents
        self._mixer_hidden_size = mixer_hidden_size

        input_dims = input_dims * num_agents
        self._hyper_w_1 = nn.Sequential(
            nn.Linear(input_dims, mixer_hidden_size),
            activation_function,
            nn.Linear(mixer_hidden_size, mixer_hidden_size * num_agents)
        ).to(device)

        self._hyper_w_2 = nn.Sequential(
            nn.Linear(input_dims, mixer_hidden_size),
            activation_function,
            nn.Linear(mixer_hidden_size, mixer_hidden_size)
        ).to(device)

        self._hyper_b_1 = nn.Linear(input_dims, mixer_hidden_size).to(device)
        self._hyper_b_2 = nn.Sequential(
            nn.Linear(input_dims, mixer_hidden_size),
            activation_function,
            nn.Linear(mixer_hidden_size, 1)
        ).to(device)

        self.to(device)

    def get_argmax_action(self, states, agent_ids):
        return self._q_network(states, agent_ids).argmax(dim=-1)
    
    def get_q_values(self, states, agent_ids):
        return self._q_network(states, agent_ids)

    def forward(self, states, agent_ids):
        values = self._q_network(states, agent_ids)      
        values = values.max(dim=-1)[0].unsqueeze(-1)
        values = values.reshape(-1, 1, self._num_agents)

        states = states.reshape(-1, self._num_agents * states.shape[-1])
        w1 = self._hyper_w_1(states)
        w1 = torch.abs(w1) # absolute activation function, ensure that the mixing network weights are non-negative
        w1 = w1.view(-1, self._num_agents, self._mixer_hidden_size)
        b1 = self._hyper_b_1(states)
        b1 = b1.view(-1, 1, self._mixer_hidden_size)
        hidden = F.elu(torch.bmm(values, w1) + b1)

        w2 = self._hyper_w_2(states)
        w2 = torch.abs(w2)
        w2 = w2.view(-1, self._mixer_hidden_size, 1)
        b2 = self._hyper_b_2(states)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(-1, 1)
        return q_total
