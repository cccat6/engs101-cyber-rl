# ENGS 101 Cyber RL Final Project

The official submission site is [arena.codefly.run](https://arena.codefly.run).
The project has two separately recorded stages: an Intermediate submission due
November 1, 2026 at 23:59 Eastern Time, and a Final submission due November 17,
2026 at 23:59 Eastern Time. Select the correct stage on the upload page and use
your own seven-character student ID.

This repository contains the student starter and a small local environment.
The full [project handout (PDF)](docs/final_project.pdf) and its
[LaTeX source](docs/final_project.tex) are included in the repository.

Students only need to use:

- `student_agent.py` - implement the agent;
- `run.py` - train and test it locally;
- `scenarios/base.yaml` - the public environment.

The code under `src/` is the local backend and does not need to be modified.

## Setup

Use Python 3.11 or later (`StrEnum` is unavailable in Python 3.10).
Clone this public repository and run commands inside the cloned folder:

```bash
git clone https://github.com/cccat6/engs101-cyber-rl.git
cd engs101-cyber-rl
python --version
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
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

The receipt only means the ZIP format was accepted. The server then imports
the code, checks parameters/actions, trains both roles, saves/reloads weights
and evaluates. Fix any `error` status and resubmit. A working learner can score
poorly; that is different from a failed run.

Intermediate points are **60 / 80 / 100** when zero / one / both roles beat the
original teacher. A baseline-centered role score must be strictly above 50;
ties do not count. Final opens after the course freezes the stronger teacher.
Final points are 100 for two role wins and 80 for one. With no Final-teacher
wins, points are 60 / 40 / 20 if both / one / neither role beats the original
teacher. Both comparisons use the same final student checkpoints. Cross-play
rankings are diagnostic and do not determine the stage grade.

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

After improving and testing your agent, create the source-only ZIP:

```bash
python package_submission.py YOUR_ID
# Include helpers imported by student_agent.py when needed:
python package_submission.py YOUR_ID --include helper.py
```

Replace `YOUR_ID` with your own seven-character Dartmouth ID (e.g. `f001abc`).
Upload the resulting ZIP to https://arena.codefly.run and select **Intermediate**.
No GitHub account or private staff code is needed. Do not upload the full clone.
Use your own ID honestly and keep the upload URL within the class. Each accepted
version is archived with an Eastern timestamp; the latest upload is the active
entry. The deadline uses receipt time, not the time the grading queue finishes.
Both deadlines are 11:59 PM America/New_York (EST/UTC-5 on November 1 and 17).

Upload one ZIP named with your seven-character student ID, for example
`f001abc.zip`. The ZIP may contain only:

- `run.py`;
- `student_agent.py`;
- optional helper modules ending in `.py`.

The files may be at the ZIP root or inside one folder whose name matches the
student ID. The ZIP and its extracted Python source must each be at most
100 KiB. Do not include this repository, scenarios, dependencies, checkpoints,
pretrained weights, or generated files. The server supplies the environment
and trains fresh Red and Blue models from the submitted source. Submissions
that do not follow this format may be rejected or removed from the submission
system; upload a corrected archive using your own seven-character student ID.
