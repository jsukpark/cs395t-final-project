# cs395t-final-project
Final Project for CS395T/CSE392: Predictive Machine Learning.


## Installation

1. **Set up python virtual environment.** Run:
```
conda create -n py310 python=3.10  # type 'y' when conda asks you to proceed
conda activate py310
python3 -m venv .env
conda deactivate
source .env/bin/activate
```

2. **Install dependencies.** Run:
```
pip install --upgrade pip
pip install gymnasium minari stable-baselines3 torchrl
pip install git+https://github.com/google/trajax
```
