---
title: ERTriageEnv
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - healthcare
  - medical-ai
  - triage
---

# ERTriageEnv — Hospital ER Triage RL Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI agent acts as an emergency room triage nurse. Each step it observes a waiting room of patients described by clinical vitals and must decide **which patient to admit to treatment next**. The environment rewards correct clinical prioritisation and penalises preventable deaths.

**Real-world grounding:** severity scoring is derived from the [NEWS2 (National Early Warning Score 2)](https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2) system used in UK NHS hospitals.

---

## Motivation

ER triage is a high-stakes decision task performed under time pressure with incomplete information. Clinicians must simultaneously assess multiple patients, account for deterioration rates and comorbidities, and allocate scarce resources. This environment models those exact constraints — making it a meaningful benchmark for risk-assessment and prioritisation agents that goes far beyond toy problems.

---

## Environment description

At each step the agent receives a variable-length queue of patients with 11 clinical features each. The agent selects one to admit to an available bed (or holds). Patients left waiting deteriorate — some die if delayed too long. The episode ends after `max_steps` or when the queue clears.

---

## Three tasks (easy → medium → hard)

| | Easy | Medium | Hard |
|---|---|---|---|
| Queue size | 10 | 20 | 30 |
| Beds | 3 | 5 | 8 |
| Max steps | 20 | 40 | 60 |
| Comorbidities | No | Yes | Yes |
| Noisy vitals | No | No | Yes |
| Surge events | No | No | Yes (every 10 steps) |
| Bed timeouts | No | No | Yes |

**Easy:** Learn the direct mapping from vitals to severity. No confounders.  
**Medium:** Comorbidities amplify risk non-linearly. Some patients deteriorate rapidly.  
**Hard:** Vitals carry measurement noise. Random patient surges stress capacity. Reason under uncertainty.

---

## Action space

```
POST /step  {"patient_index": int}
```

| Value | Meaning |
|---|---|
| `0..len(patients)-1` | Admit that patient |
| `-1` | Hold queue (penalised when beds are free) |

**Typed model:** `ERTriageAction(patient_index: int)` — `pydantic.BaseModel`

---

## Observation space

`ERTriageObservation` (Pydantic) — returned by `reset()` and inside each `StepResult`:

| Field | Type | Description |
|---|---|---|
| `patients` | `List[PatientVitals]` | Current waiting room |
| `beds_available` | `int` | Free treatment beds |
| `step` | `int` | Current step number |
| `task` | `str` | `easy` / `medium` / `hard` |
| `done` | `bool` | Episode ended |
| `reward` | `ERTriageReward?` | Full reward breakdown (null after reset) |
| `info` | `dict` | `{admitted, deaths}` |

`PatientVitals` fields (11 clinical features):

| Field | Normal range | Clinical significance |
|---|---|---|
| `heart_rate` | 60–100 bpm | Tachycardia / bradycardia |
| `bp_systolic` | 90–140 mmHg | Shock < 80 |
| `spo2` | 95–100 % | Hypoxia < 90 |
| `respiratory_rate` | 12–20 /min | Failure < 8 or > 30 |
| `temperature` | 36.1–37.2 °C | Sepsis > 38.5 |
| `pain_score` | 0–10 | Self-reported |
| `age` | — | Elderly = amplified risk |
| `wait_minutes` | — | Minutes in queue |
| `deterioration_rate` | — | Severity increase per step |
| `comorbidities` | 0–3 | Chronic conditions (medium/hard) |
| `severity_noisy` | 0.0–1.0 | Estimated risk (noisy in hard) |

---

## Reward function

`ERTriageReward` — typed Pydantic model returned with every `step()`:

| Component | Signal | Purpose |
|---|---|---|
| `severity_saved` | `+severity_true × 10` | Reward proportional to criticality |
| `priority_bonus` | `+2` | Bonus for choosing the most critical patient |
| `wait_penalty` | `−0.5 × total_wait / 60` | Dense signal — accumulates every step |
| `death_penalty` | `−20` per death | Largest penalty — preventable death |
| `hold_penalty` | `−5` | Holding when beds are free |

The reward is **dense** — every step has a non-zero signal due to the wait penalty. The agent always knows whether it is making progress.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `{"status": "healthy"}` |
| `GET` | `/metadata` | `{"name": str, "description": str, ...}` |
| `GET` | `/schema` | `{"action":{}, "observation":{}, "state":{}}` |
| `POST` | `/mcp` | JSON-RPC 2.0 tool interface |
| `GET` | `/tasks` | List all 3 tasks with config |
| `POST` | `/reset` | `{"task": "easy", "seed": 42}` → `ERTriageObservation` |
| `POST` | `/step` | `{"patient_index": 0}` → `StepResult` |
| `GET` | `/state` | `ERTriageState` (no advance) |
| `POST` | `/grade` | Score action sequence → `0.0–1.0` |

`StepResult` shape:
```json
{
  "observation": { "patients": [...], "beds_available": 2, "step": 1, ... },
  "reward": { "total": 6.32, "severity_saved": 8.5, "priority_bonus": 2.0,
              "wait_penalty": -3.68, "death_penalty": 0.0, "hold_penalty": 0.0 },
  "done": false,
  "info": { "admitted": 1, "deaths": 0 }
}
```

---

## Grading (0.0–1.0)

```
POST /grade  {"task": "easy", "actions": [2, 0, 5, ...]}
```

Runs 10 deterministic episodes (fixed seeds 100–109):

| Component | Weight | Description |
|---|---|---|
| Deaths avoided | **0.50** | Normalised vs worst-case baseline |
| Priority accuracy | **0.30** | Near-optimal patient chosen per step |
| Queue efficiency | **0.20** | Low hold-when-free rate |

---

## Baseline scores

| Task | Greedy agent | Random agent | FIFO agent |
|---|---|---|---|
| Easy | **0.8429** | 0.5214 | 0.4786 |
| Medium | **0.6000** | 0.2969 | 0.3000 |
| Hard | **0.7023** | 0.4786 | 0.4875 |

---

## Setup and usage

### Docker

```bash
docker build -t er-triage-env .
docker run -p 7860:7860 er-triage-env

# Test all openenv runtime criteria
curl http://localhost:7860/health          # {"status": "healthy"}
curl http://localhost:7860/metadata        # {"name": "er_triage_env", ...}
curl http://localhost:7860/schema          # {"action":{}, "observation":{}, "state":{}}
curl -X POST http://localhost:7860/mcp -H 'Content-Type: application/json' -d '{}'
curl -X POST http://localhost:7860/reset   -H 'Content-Type: application/json' -d '{}'
```

### Local Python

```bash
pip install -r requirements.txt

# Start server
uvicorn server.app:app --port 7860

# OR via uv
uv run server
```

### Run inference (LLM agent)

```bash
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini
python inference.py
```

### Run baseline without API key

```bash
# Reproducible greedy baseline — no API key required
python inference.py --greedy
```

### Validate

```bash
# OpenEnv CLI validator (local structure check)
openenv validate .

# Full test suite (77 tests, including openenv runtime criteria)
PYTHONPATH=src:. pytest tests/ -v

# Custom validator (51 checks)
PYTHONPATH=src:. python validate.py
```

---

## Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/spaces with **Docker** SDK
2. Clone the Space repo and copy this project into it:

```bash
git clone https://huggingface.co/spaces/<your-username>/<space-name>
cp -r er_triage_env/* <space-name>/
cd <space-name>
git add .
git commit -m "Add ERTriageEnv"
git push
```

3. Set Space secrets (Settings → Variables and secrets):
   - `OPENAI_API_KEY` — your OpenAI key
   - `MODEL_NAME` — e.g. `gpt-4o-mini`

4. After deployment, validate the live Space:

```bash
openenv validate https://<your-username>-<space-name>.hf.space
```

---

## Project structure

```
er_triage_env/
├── inference.py                     ← LLM agent entry point (run this for grading)
├── validate.py                      ← Pre-submission validator (51 checks)
├── openenv.yaml                     ← OpenEnv spec
├── pyproject.toml                   ← Package config + server entry point
├── uv.lock                          ← Reproducible dependency lock
├── Dockerfile                       ← HF Spaces compatible (port 7860, non-root)
├── requirements.txt
├── scores.json                      ← Pre-computed greedy baseline scores
├── demo.py                          ← 4-agent comparison demo
│
├── server/                          ← OpenEnv-spec entry point
│   ├── app.py                       ← FastAPI server (all endpoints incl. /health /schema /mcp)
│   ├── models.py                    ← openenv-core Action/Observation/State types
│   └── environment.py               ← openenv-core Environment wrapper
│
└── src/                             ← Simulation engine
    └── envs/er_triage/
        ├── models.py                ← Rich Pydantic models (ERTriageReward, StepResult)
        ├── graders/graders.py       ← Deterministic 0.0-1.0 graders
        └── server/
            ├── environment.py       ← Full simulation engine (NEWS2 scoring)
            └── app.py               ← Simulation HTTP API
```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | For LLM | — | Your OpenAI API key |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model identifier |
| `API_BASE_URL` | No | OpenAI | API base URL |
| `ER_TASK` | No | `easy` | Default task for server |
| `PORT` | No | `7860` | Server port |
