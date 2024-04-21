#!/usr/bin/env python3

# license TBD

# system imports

# library imports
import torch
from torch import nn
import numpy as np

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


class BaseMultiAgentMLP(nn.Module):

    def __init__(self,
        input_dims: int = 1,
        representation_layer_dims: int = 64, 
        hidden_cells: list = [256, 256],
        out_dims: int = 1,
        device = torch.device("cuda:0"),
        activation_function = nn.ReLU(),
        model_has_dropout: bool = False,
        dropout_rate: float = 0.1,
        has_lstm: bool = False,
        has_attention: bool = False,
        num_agents: int = 2,
        ) -> None:
        super().__init__()
        self._device = device
        self._activation_function = activation_function
        
        self._representation_layer = nn.Linear(input_dims, representation_layer_dims, device=self._device)
        self._representation_layer_dims = representation_layer_dims
        self._representation_layer_activation = self._activation_function

        if model_has_dropout:
            self._representation_layer = nn.Sequential(
                self._representation_layer,
                self._representation_layer_activation,
                nn.Dropout(p=dropout_rate)
            )
        else:
            self._representation_layer = nn.Sequential(
                self._representation_layer,
                self._representation_layer_activation,
            )
        
        self._has_lstm = has_lstm
        if has_lstm and not has_attention:
            lstm_layer = nn.LSTM(representation_layer_dims, representation_layer_dims)
            self._lstm_layer = lstm_layer

        if has_attention and not has_lstm:
            assert representation_layer_dims % 8 == 0, "The representation layer dimensions must be divisible by 8 for the attention layer to work."
            attention_layer = nn.TransformerEncoderLayer(d_model=representation_layer_dims, nhead=8)
            self._has_attention = has_attention
            self._representation_layer.append(attention_layer)

        self._representation_layer.apply(init_weights)
        
        input_dims = representation_layer_dims + num_agents
        self._hidden_and_outer_layers = []
        for i in range(len(hidden_cells)):
            if i == 0:
                self._hidden_and_outer_layers.append(nn.Linear(input_dims, hidden_cells[i], device=self._device))
            else:
                self._hidden_and_outer_layers.append(nn.Linear(hidden_cells[i-1], hidden_cells[i], device=self._device))
            
            if model_has_dropout:
                self._hidden_and_outer_layers.append(nn.Dropout(p=dropout_rate))
            self._hidden_and_outer_layers.append(self._activation_function)
            input_dims = hidden_cells[i]

        self._hidden_and_outer_layers.append(nn.Linear(hidden_cells[-1], out_dims, device=self._device)) 
        self._hidden_and_outer_layers = nn.Sequential(*self._hidden_and_outer_layers)
        self._hidden_and_outer_layers.apply(init_weights)
        self.to(self._device)


class DiscreteLinearMultiAgentCritic(BaseMultiAgentMLP):
    def __init__(self,
        input_dims: int = 1,
        hidden_cells: list = [256, 256],
        out_dims: int = 1,
        device = torch.device("cuda:0"),
        activation_function = nn.ReLU(),
        model_has_dropout: bool = False,
        dropout_rate: float = 0.1,
        has_lstm: bool = False,
        has_attention: bool = False,
        num_agents: int = 2,
        representation_layer_dims: int = 64,
        ) -> None:
        
        super().__init__(
            input_dims=input_dims,
            hidden_cells=hidden_cells,
            out_dims=out_dims,
            device=device,
            activation_function=activation_function,
            model_has_dropout=model_has_dropout,
            dropout_rate=dropout_rate,
            has_lstm=has_lstm,
            has_attention=has_attention,
            num_agents=num_agents,
            representation_layer_dims=representation_layer_dims,
        )

    def forward(self, state, agent_ids):    
        representation = self._representation_layer(state)  

        if self._has_lstm:
            h0 = torch.randn(1, state.shape[1], self._representation_layer_dims).to(self._device)
            c0 = torch.randn(1, state.shape[1], self._representation_layer_dims).to(self._device)
            representation, (ht, ct) = self._lstm_layer(representation, (h0, c0))

        hidden_input = torch.concat([representation, agent_ids], dim=-1)   
        q_value = self._hidden_and_outer_layers(hidden_input)
        
        return q_value
