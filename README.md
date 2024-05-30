# Search and Rescue 2D
## About
The code contained in this repository is the original implementation of the article: 
"Causal Reinforcement Learning for Optimisation of Robot Dynamics in Unknown Environments"

## Installation
    conda env create -f environment.yaml

## Training 
When training for comparison of Causal and Non-Causal models and training in parallel then a separate repo directory is required for each.

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
