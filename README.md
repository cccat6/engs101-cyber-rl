# ENGS 101 Cyber RL Final Project

This repository contains the student starter and a small local environment.

Students only need to use:

- `student_agent.py` - implement the agent;
- `run.py` - train and test it locally;
- `scenarios/base.yaml` - the public environment.

The code under `src/` is the local backend and does not need to be modified.

## Setup

Python 3.11 or later is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The requirements are PyTorch, NumPy, and PyYAML. Student agent code may use
the Python standard library, PyTorch, and NumPy. Existing RL implementations
such as Stable-Baselines3 and RLlib are not allowed.

## Run

```powershell
python run.py
```

The command trains the same code as Red and Blue, saves both models, reloads
them, and runs local evaluation.

Optional local settings are:

```powershell
python run.py --steps 5000 --eval-episodes 10 --device cuda
```

The server supplies the official training budget, seeds, environment, and
device.

## Agent interface

Keep this function in `student_agent.py`:

```python
def create_agent(config):
    return YourAgent(config)
```

The returned object must provide:

```python
agent.model
agent.reset(seed)
agent.act(observation, training)
agent.learn(transition)
```

The included `SimpleAgent` is a minimal one-layer policy-gradient example. It
runs end to end and updates its model, but it is not intended to be competitive.
