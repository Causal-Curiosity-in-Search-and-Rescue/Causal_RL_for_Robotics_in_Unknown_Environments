# Search and Resuce 2D

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

- update the model_path in the config.json 

```bash
python predict.py config.json
```