# Search and Rescue 2D
## About
The code contained in this repository is the original implementation of the article: 
"Causal Reinforcement Learning for Optimisation of Robot Dynamics in Unknown Environments"

## Installation
    conda env create -f environment.yaml

## Training 

### Usage for Causal

```bash
python train.py config.json -causal
```

### Usage for NonCausal

```bash
python train.py config.json 
```

## Prediction 

- update the model_path in the config.json and select which environment to use 

```bash
python predict.py config.json
```
