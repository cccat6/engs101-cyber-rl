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

For each role, the runner uses one frozen, course-owned teacher policy as the
opponent. The public repository contains teacher checkpoints for local work;
the cloud uses separately trained private checkpoints with the same interface.
The teacher opponent never learns or changes during a student training run.

The official stable score is relative to the course baseline. The baseline is
centered at 50 separately for Red and Blue, and the two relative role scores
are averaged. The full student Red-by-Blue matrix is reported separately.

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

`agent.model` must be a `torch.nn.Module` with at most 1,000,000 registered
parameters per role. The local runner performs the same check as the
course server and prints the saved model size after each run.

The included `SimpleAgent` is a minimal one-layer policy-gradient example. It
runs end to end and updates its model, but it is not intended to be competitive.

## Submission

Upload one ZIP named with your seven-character student ID, for example
`f001abc.zip`. The ZIP may contain only:

- `run.py`;
- `student_agent.py`;
- optional helper modules ending in `.py`.

The files may be at the ZIP root or inside one folder whose name matches the
student ID. The ZIP and its extracted Python source must each be at most
100 KiB. Do not include this repository, scenarios, dependencies, checkpoints,
pretrained weights, or generated files. The server supplies the environment
and trains fresh Red and Blue models from the submitted source.
