## QMIX with LLM based exploration

This project is developed as a part of the DS-5983 Large language models course taught at Northeastern University in Spring, 2024. The project is based on the QMIX algorithm and the exploration is done using the large language model (LLM) based exploration. The project is implemented using the Python, torch, torchrl, huggingface and vicuna-7b fastchat library.

### Install

```bash
pip install -r requirements.txt
```

### Usage

To training the QMIX without lstm or attention, run the following command:
```bash
python <path/to/large-language-models-project>/llm_marl/multi_agent_qmix_discrete_env.py --config <path/to/large-language-models-project>/llm_marl/config/qmix/multi_agent_simple_spread.yaml
```

To training the QMIX with lstm, run the following command:

```bash
python <path/to/large-language-models-project>/llm_marl/multi_agent_qmix_discrete_env.py --config <path/to/large-language-models-project>/llm_marl/config/qmix/multi_agent_simple_spread_lstm.yaml
```

To training the QMIX with attention, run the following command:

```bash
python <path/to/large-language-models-project>/llm_marl/multi_agent_qmix_discrete_env.py --config <path/to/large-language-models-project>/llm_marl/config/qmix/multi_agent_simple_spread_attention.yaml
```

To training the QMIX with attention and astar oracle, run the following command:

```bash
python <path/to/large-language-models-project>/llm_marl/multi_agent_qmix_discrete_env.py --config <path/to/large-language-models-project>/llm_marl/config/qmix/multi_agent_simple_spread_astar.yaml
```

To training the QMIX with attention and vicuna-7b based exploration, run the following command:

```bash
python <path/to/large-language-models-project>/llm_marl/multi_agent_qmix_discrete_env.py --config <path/to/large-language-models-project>/llm_marl/config/qmix/multi_agent_simple_spread_llm.yaml
```

To training the QMIX with attention and fine-tuned vicuna-7b based exploration, run the following command:

```bash
python <path/to/large-language-models-project>/llm_marl/multi_agent_qmix_discrete_env.py --config <path/to/large-language-models-project>/llm_marl/config/qmix/multi_agent_simple_spread_fine_tune_llm.yaml
```