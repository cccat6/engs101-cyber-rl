# ENGS 101 Cyber RL Final Project

This public repository contains:

- `student_agent.py`: the only file students need to edit;
- `scenarios/`: the small public environment;
- `cyberrl_eval`: the local training and testing runner;
- `assignment/final_project.tex`: the project handout.

The cloud system uses the same agent interface with a hidden environment.

## Setup

Python 3.11 or later is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Student code may use the Python standard library, PyTorch, and NumPy. Do not
use existing RL implementations such as Stable-Baselines3 or RLlib.

## Implement the agent

Keep this function in `student_agent.py`:

```python
def create_agent(config):
    return YourAgent(config)
```

The returned object needs four things:

```python
agent.model                         # torch.nn.Module
agent.reset(seed)                   # called at the start of each episode
agent.act(observation, training)    # returns one valid integer action
agent.learn(transition)             # called once per training step
```

`config` contains only:

```text
role, observation_dim, num_actions, total_steps, seed, device
```

Use `config.device` when creating tensors and moving the model. The server
selects the CUDA device. Local runs may use `auto`, `cpu`, `cuda`, `xpu`, or
`mps` (`metal` is accepted as an alias).

`observation` contains:

```text
observation.features       float32 vector
observation.action_mask    boolean vector of valid actions
```

`transition` contains:

```text
observation, action, reward, next_observation, done
```

The provided agent runs but does not learn.

## Run locally

From the repository root:

```powershell
python -m cyberrl_eval.submission_cli --steps 1000 --device auto
```

This command trains the same code as Red and Blue, saves
`checkpoints/red.pt` and `checkpoints/blue.pt`, reloads both checkpoints, and
tests them on the public scenarios.

Local settings are for development only. The cloud runner supplies the
official step budget, seeds, environment, device, and evaluation episodes.

## Tests

```powershell
python -m unittest discover -s tests -v
```
