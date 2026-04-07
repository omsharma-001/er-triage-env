#!/usr/bin/env python3
"""
validate.py — Pre-submission validator for ERTriageEnv.

Run from project root:
  cd er_triage_env
  PYTHONPATH=src python validate.py

Exit 0 = ready to submit. Exit 1 = fix failures first.
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

BOLD  = "\033[1m"
GREEN = "\033[32m"
RED   = "\033[31m"
RESET = "\033[0m"

_checks = []

def check(name, category=""):
    def decorator(fn):
        _checks.append((name, category, fn))
        return fn
    return decorator

def run_all():
    passed, failed = [], []
    categories = {}
    for name, cat, fn in _checks:
        categories.setdefault(cat, []).append((name, fn))

    print(f"\n{BOLD}ERTriageEnv — Pre-submission validator{RESET}")
    print("=" * 60)

    for cat, checks in categories.items():
        if cat:
            print(f"\n{BOLD}{cat}{RESET}")
        for name, fn in checks:
            try:
                msg = fn()
                passed.append(name)
                detail = f" — {msg}" if msg else ""
                print(f"  {GREEN}PASS{RESET}  {name}{detail}")
            except Exception as e:
                failed.append((name, str(e)))
                print(f"  {RED}FAIL{RESET}  {name}")
                print(f"         {RED}{e}{RESET}")

    print("\n" + "=" * 60)
    total = len(passed) + len(failed)
    print(f"{BOLD}Results: {len(passed)}/{total} checks passed{RESET}")
    if failed:
        print(f"\n{RED}Failing checks:{RESET}")
        for name, err in failed:
            print(f"  • {name}: {err}")
        print(f"\n{RED}Fix before submitting.{RESET}")
        return False
    print(f"\n{GREEN}All checks passed! Ready to submit.{RESET}")
    return True


# ── File structure ────────────────────────────────────────────────────────────

@check("inference.py at root", "File structure")
def _():
    assert os.path.exists("inference.py")
    return "✓"

@check("openenv.yaml exists", "File structure")
def _():
    assert os.path.exists("openenv.yaml")
    return "✓"

@check("Dockerfile exists", "File structure")
def _():
    assert os.path.exists("Dockerfile")
    return "✓"

@check("requirements.txt has fastapi + openai", "File structure")
def _():
    with open("requirements.txt") as f:
        r = f.read()
    assert "fastapi" in r and "openai" in r
    return "fastapi, openai ✓"

@check("README.md has HF Spaces header + openenv tag", "File structure")
def _():
    with open("README.md") as f:
        content = f.read()
    assert "sdk: docker" in content, "missing 'sdk: docker'"
    assert "openenv" in content,     "missing 'openenv' tag"
    return "sdk:docker + openenv tag ✓"

@check("src/ __init__.py files complete", "File structure")
def _():
    for p in ["src/__init__.py", "src/core/__init__.py",
               "src/envs/__init__.py", "src/envs/er_triage/__init__.py"]:
        assert os.path.exists(p), f"missing {p}"
    return "all __init__.py ✓"


# ── OpenEnv spec ──────────────────────────────────────────────────────────────

@check("openenv.yaml valid YAML", "OpenEnv spec")
def _():
    import yaml
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    assert isinstance(spec, dict)
    return "valid ✓"

@check("openenv.yaml has 3+ tasks", "OpenEnv spec")
def _():
    import yaml
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    tasks = spec.get("tasks", [])
    assert len(tasks) >= 3, f"need 3+, found {len(tasks)}"
    return f"{len(tasks)} tasks: {[t['name'] for t in tasks]}"

@check("openenv.yaml has grading config with score_range", "OpenEnv spec")
def _():
    import yaml
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    g = spec.get("grading", {})
    assert "score_range" in g
    assert g["score_range"] == [0.0, 1.0]
    return f"score_range={g['score_range']}, episodes={g.get('episodes')}"

@check("openenv.yaml has Pydantic model schemas", "OpenEnv spec")
def _():
    import yaml
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    models = spec.get("models", {})
    assert "action" in models,      "missing models.action"
    assert "observation" in models, "missing models.observation"
    assert "reward" in models,      "missing models.reward"
    for key in ["action", "observation", "reward"]:
        assert models[key].get("type") == "pydantic.BaseModel", \
            f"models.{key}.type must be 'pydantic.BaseModel'"
    return "action, observation, reward all Pydantic ✓"

@check("openenv.yaml has OPENAI_API_KEY in inference.credentials", "OpenEnv spec")
def _():
    import yaml
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    creds = spec.get("inference", {}).get("credentials", [])
    assert "OPENAI_API_KEY" in creds, f"OPENAI_API_KEY missing from credentials: {creds}"
    return f"credentials: {creds}"


# ── Pydantic models ───────────────────────────────────────────────────────────

@check("All models are pydantic.BaseModel subclasses", "Pydantic models")
def _():
    import pydantic
    from envs.er_triage.models import (
        ERTriageAction, ERTriageObservation, ERTriageReward,
        ERTriageState, StepResult, PatientVitals,
    )
    for cls in [ERTriageAction, ERTriageObservation, ERTriageReward,
                ERTriageState, StepResult, PatientVitals]:
        assert issubclass(cls, pydantic.BaseModel), f"{cls.__name__} not Pydantic"
    return "Action, Observation, Reward, State, StepResult, PatientVitals ✓"

@check("ERTriageAction has patient_index: int", "Pydantic models")
def _():
    from envs.er_triage.models import ERTriageAction
    a = ERTriageAction(patient_index=3)
    assert a.patient_index == 3
    a2 = ERTriageAction(patient_index=-1)
    assert a2.patient_index == -1
    return "patient_index=3 and -1 ✓"

@check("ERTriageReward has total + 5 breakdown fields", "Pydantic models")
def _():
    from envs.er_triage.models import ERTriageReward
    fields = set(ERTriageReward.model_fields.keys())
    required = {"total", "severity_saved", "priority_bonus",
                "wait_penalty", "death_penalty", "hold_penalty"}
    missing = required - fields
    assert not missing, f"missing: {missing}"
    r = ERTriageReward(total=-3.5, severity_saved=10.0, priority_bonus=2.0,
                       wait_penalty=-5.5, death_penalty=0.0, hold_penalty=0.0)
    assert r.total == -3.5
    return f"6 fields ✓ (total, severity_saved, priority_bonus, wait_penalty, death_penalty, hold_penalty)"

@check("ERTriageObservation.reward is ERTriageReward (not float)", "Pydantic models")
def _():
    from envs.er_triage.models import ERTriageObservation, ERTriageReward
    import typing
    ann = ERTriageObservation.model_fields["reward"].annotation
    ann_str = str(ann)
    assert "ERTriageReward" in ann_str or "float" not in ann_str, \
        f"reward field should be ERTriageReward, got {ann_str}"
    return f"reward: Optional[ERTriageReward] ✓"

@check("StepResult has observation, reward, done, info", "Pydantic models")
def _():
    from envs.er_triage.models import StepResult
    fields = set(StepResult.model_fields.keys())
    for f in ["observation", "reward", "done", "info"]:
        assert f in fields, f"StepResult missing '{f}'"
    return "observation, reward, done, info ✓"

@check("PatientVitals has 11+ clinical fields", "Pydantic models")
def _():
    from envs.er_triage.models import PatientVitals
    fields = set(PatientVitals.model_fields.keys())
    required = {"heart_rate", "bp_systolic", "spo2", "respiratory_rate",
                "temperature", "pain_score", "age", "wait_minutes",
                "deterioration_rate", "comorbidities", "severity_noisy"}
    missing = required - fields
    assert not missing, f"missing: {missing}"
    return f"{len(fields)} fields ✓"

@check("JSON round-trip: StepResult serialises cleanly", "Pydantic models")
def _():
    from envs.er_triage.server.environment import ERTriageEnvironment
    from envs.er_triage.models import ERTriageAction, StepResult
    env = ERTriageEnvironment(task="easy", seed=0)
    env.reset()
    result = env.step(ERTriageAction(patient_index=0))
    j = result.model_dump_json()
    r2 = StepResult.model_validate_json(j)
    assert abs(r2.reward.total - result.reward.total) < 0.001
    return f"StepResult ↔ JSON ✓ (reward.total={result.reward.total:.3f})"


# ── Environment API ───────────────────────────────────────────────────────────

@check("reset() returns ERTriageObservation(step=0, reward=None)", "Environment API")
def _():
    from envs.er_triage.server.environment import ERTriageEnvironment
    from envs.er_triage.models import ERTriageObservation
    env = ERTriageEnvironment(task="easy", seed=42)
    obs = env.reset()
    assert isinstance(obs, ERTriageObservation)
    assert obs.step == 0
    assert obs.reward is None
    assert obs.done is False
    assert len(obs.patients) == 10
    return f"step=0, {len(obs.patients)} patients, reward=None ✓"

@check("step() returns StepResult with typed ERTriageReward", "Environment API")
def _():
    from envs.er_triage.server.environment import ERTriageEnvironment
    from envs.er_triage.models import ERTriageAction, StepResult, ERTriageReward
    env = ERTriageEnvironment(task="easy", seed=0)
    env.reset()
    result = env.step(ERTriageAction(patient_index=0))
    assert isinstance(result, StepResult),            "step() must return StepResult"
    assert isinstance(result.reward, ERTriageReward), "reward must be ERTriageReward"
    assert result.observation.step == 1,              "observation.step should be 1"
    assert result.done is False or result.done is True
    return f"StepResult(reward.total={result.reward.total:.2f}, step={result.observation.step}) ✓"

@check("reward breakdown components sum to total", "Environment API")
def _():
    from envs.er_triage.server.environment import ERTriageEnvironment
    from envs.er_triage.models import ERTriageAction
    env = ERTriageEnvironment(task="medium", seed=5)
    env.reset()
    for i in range(5):
        result = env.step(ERTriageAction(patient_index=0 if env._queue else -1))
        r = result.reward
        expected = r.severity_saved + r.priority_bonus + r.wait_penalty + r.death_penalty + r.hold_penalty
        assert abs(r.total - expected) < 0.01, f"total={r.total} != sum={expected}"
    return "5 steps: total == Σ components ✓"

@check("state() returns ERTriageState Pydantic model", "Environment API")
def _():
    from envs.er_triage.server.environment import ERTriageEnvironment
    from envs.er_triage.models import ERTriageState
    import pydantic
    env = ERTriageEnvironment(task="medium", seed=1)
    env.reset()
    s = env.state
    assert isinstance(s, pydantic.BaseModel)
    assert s.task == "medium"
    assert s.episode_id is not None
    return f"ERTriageState(episode={s.episode_id}, task={s.task}) ✓"

@check("Reward is non-sparse: varies across steps", "Environment API")
def _():
    from envs.er_triage.server.environment import ERTriageEnvironment
    from envs.er_triage.models import ERTriageAction
    env = ERTriageEnvironment(task="medium", seed=7)
    env.reset()
    totals = []
    for i in range(8):
        result = env.step(ERTriageAction(patient_index=i % 4))
        totals.append(round(result.reward.total, 1))
        if result.done: break
    unique = len(set(totals))
    assert unique >= 2, f"reward not varying: {totals}"
    return f"{unique} unique reward values in {len(totals)} steps ✓"

@check("Reproducibility: same seed → identical episode", "Environment API")
def _():
    from envs.er_triage.server.environment import ERTriageEnvironment
    env1 = ERTriageEnvironment(task="hard", seed=123)
    env2 = ERTriageEnvironment(task="hard", seed=123)
    obs1 = env1.reset()
    obs2 = env2.reset()
    assert len(obs1.patients) == len(obs2.patients)
    for p1, p2 in zip(obs1.patients, obs2.patients):
        assert abs(p1.heart_rate - p2.heart_rate) < 0.01
    return "hard seed=123: identical ✓"

@check("All 3 tasks terminate within max_steps", "Environment API")
def _():
    from envs.er_triage.server.environment import ERTriageEnvironment, TASK_CONFIG
    from envs.er_triage.models import ERTriageAction
    for task in ["easy", "medium", "hard"]:
        env = ERTriageEnvironment(task=task, seed=99)
        env.reset()
        steps, done = 0, False
        while not done and steps <= TASK_CONFIG[task]["max_steps"] + 2:
            result = env.step(ERTriageAction(patient_index=-1))
            done, steps = result.done, steps + 1
        assert done, f"{task}: never terminated"
    return "easy/medium/hard all terminate ✓"


# ── Graders ───────────────────────────────────────────────────────────────────

@check("Graders produce [0.0, 1.0] scores for all 3 tasks", "Graders")
def _():
    from envs.er_triage.graders.graders import grade, greedy_agent
    scores = {}
    for task in ["easy", "medium", "hard"]:
        r = grade(task, greedy_agent)
        assert 0.0 <= r["score"] <= 1.0, f"{task}={r['score']} out of range"
        scores[task] = r["score"]
    return str({t: round(s,4) for t,s in scores.items()})

@check("Graders are deterministic (same score twice)", "Graders")
def _():
    from envs.er_triage.graders.graders import grade, greedy_agent
    for task in ["easy", "medium", "hard"]:
        s1 = grade(task, greedy_agent)["score"]
        s2 = grade(task, greedy_agent)["score"]
        assert s1 == s2, f"{task}: {s1} != {s2}"
    return "3 tasks × 2 runs = identical ✓"

@check("Greedy agent beats random on all tasks", "Graders")
def _():
    from envs.er_triage.graders.graders import grade, greedy_agent, random_agent
    for task in ["easy", "medium", "hard"]:
        g = grade(task, greedy_agent)["score"]
        r = grade(task, random_agent)["score"]
        assert g >= r, f"[{task}] greedy {g:.4f} should >= random {r:.4f}"
    return "greedy ≥ random on easy/medium/hard ✓"

@check("Score components weight to composite correctly", "Graders")
def _():
    from envs.er_triage.graders.graders import grade, greedy_agent
    r = grade("medium", greedy_agent)
    expected = round(0.50*r["deaths_score"] + 0.30*r["priority_score"] + 0.20*r["efficiency_score"], 4)
    assert abs(r["score"] - expected) < 0.001, f"{r['score']} != {expected}"
    return f"0.50×{r['deaths_score']:.3f}+0.30×{r['priority_score']:.3f}+0.20×{r['efficiency_score']:.3f}={r['score']:.4f} ✓"


# ── HTTP server ───────────────────────────────────────────────────────────────

@check("GET /health → 200 {status: ok}", "HTTP server")
def _():
    from fastapi.testclient import TestClient
    from envs.er_triage.server.app import app
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    return "200 ✓"

@check("GET /tasks → 3 tasks with action/observation_space", "HTTP server")
def _():
    from fastapi.testclient import TestClient
    from envs.er_triage.server.app import app
    r = TestClient(app).get("/tasks")
    assert r.status_code == 200
    tasks = r.json()
    assert len(tasks) == 3
    for t in tasks:
        assert "action_space" in t and "observation_space" in t
    return f"{[t['name'] for t in tasks]} ✓"

@check("POST /reset → ERTriageObservation (reward=null, step=0)", "HTTP server")
def _():
    from fastapi.testclient import TestClient
    from envs.er_triage.server.app import app
    r = TestClient(app).post("/reset", json={"task": "easy", "seed": 42})
    assert r.status_code == 200
    obs = r.json()
    assert obs["reward"] is None
    assert obs["step"] == 0
    assert obs["task"] == "easy"
    assert len(obs["patients"]) == 10
    return "reward=null, step=0, 10 patients ✓"

@check("POST /step → StepResult {observation, reward{total,...}, done, info}", "HTTP server")
def _():
    from fastapi.testclient import TestClient
    from envs.er_triage.server.app import app
    c = TestClient(app)
    c.post("/reset", json={"task": "easy", "seed": 0})
    r = c.post("/step", json={"patient_index": 0})
    assert r.status_code == 200
    body = r.json()
    assert "observation" in body,            "missing observation"
    assert "reward" in body,                 "missing reward"
    assert "done" in body,                   "missing done"
    assert "info" in body,                   "missing info"
    assert "total" in body["reward"],        "reward missing total"
    assert "severity_saved" in body["reward"], "reward missing severity_saved"
    assert body["observation"]["step"] == 1, "observation.step should be 1"
    return f"StepResult(reward.total={body['reward']['total']}, obs.step={body['observation']['step']}) ✓"

@check("GET /state → ERTriageState after step", "HTTP server")
def _():
    from fastapi.testclient import TestClient
    from envs.er_triage.server.app import app
    c = TestClient(app)
    c.post("/reset", json={"task": "medium", "seed": 5})
    c.post("/step", json={"patient_index": 0})
    r = c.get("/state")
    assert r.status_code == 200
    s = r.json()
    assert s["admitted_count"] == 1
    assert s["step_count"] == 1
    return f"admitted={s['admitted_count']}, step={s['step_count']} ✓"

@check("POST /grade → score in [0.0, 1.0] for all tasks", "HTTP server")
def _():
    from fastapi.testclient import TestClient
    from envs.er_triage.server.app import app
    c = TestClient(app)
    for task in ["easy", "medium", "hard"]:
        r = c.post("/grade", json={"task": task, "actions": [0,1,0,2,1]})
        assert r.status_code == 200
        score = r.json()["score"]
        assert 0.0 <= score <= 1.0, f"{task} score {score} out of range"
    return "easy/medium/hard all in [0.0, 1.0] ✓"

@check("Hard mode hides severity_true from patients", "HTTP server")
def _():
    from fastapi.testclient import TestClient
    from envs.er_triage.server.app import app
    c = TestClient(app)
    r = c.post("/reset", json={"task": "hard", "seed": 1})
    assert r.status_code == 200
    patients = r.json()["patients"]
    for p in patients:
        assert p.get("severity_true") is None, "hard mode should hide severity_true"
    return f"severity_true=null for all {len(patients)} hard-mode patients ✓"


# ── Inference script ──────────────────────────────────────────────────────────

@check("inference.py uses OpenAI client", "Inference script")
def _():
    with open("inference.py") as f:
        src = f.read()
    assert "from openai import OpenAI" in src
    return "✓"

@check("inference.py reads OPENAI_API_KEY (primary credential)", "Inference script")
def _():
    with open("inference.py") as f:
        src = f.read()
    assert "OPENAI_API_KEY" in src, "OPENAI_API_KEY not in inference.py"
    return "OPENAI_API_KEY ✓"

@check("inference.py reads MODEL_NAME and API_BASE_URL", "Inference script")
def _():
    with open("inference.py") as f:
        src = f.read()
    for var in ["MODEL_NAME", "API_BASE_URL"]:
        assert var in src, f"{var} not in inference.py"
    return "MODEL_NAME, API_BASE_URL ✓"

@check("inference.py writes scores.json", "Inference script")
def _():
    with open("inference.py") as f:
        src = f.read()
    assert "scores.json" in src
    return "✓"

@check("Dockerfile: port 7860, non-root user, uvicorn", "Inference script")
def _():
    with open("Dockerfile") as f:
        df = f.read()
    assert "EXPOSE 7860" in df, "missing EXPOSE 7860"
    assert "appuser" in df,     "missing non-root user"
    assert "uvicorn" in df,     "missing uvicorn in CMD"
    return "EXPOSE 7860, appuser, uvicorn ✓"

@check("scores.json baseline file exists and is valid", "Inference script")
def _():
    assert os.path.exists("scores.json"), "scores.json not found"
    with open("scores.json") as f:
        data = json.load(f)
    assert "scores" in data
    for task in ["easy", "medium", "hard"]:
        assert task in data["scores"], f"{task} missing from scores.json"
        s = data["scores"][task]
        assert 0.0 <= s <= 1.0, f"{task} score {s} out of [0,1]"
    return f"scores: {data['scores']} ✓"


# ── checks: openenv-core CLI compliance ──────────────────────────────────────
# These mirror the exact checks run by `openenv validate .` and `openenv validate <url>`

@check("pyproject.toml exists with server entry point", "OpenEnv-core CLI")
def _():
    import tomllib
    path = "pyproject.toml"
    assert os.path.exists(path), "Missing pyproject.toml"
    with open(path, "rb") as f:
        d = tomllib.load(f)
    scripts = d.get("project", {}).get("scripts", {})
    assert "server" in scripts, f"[project.scripts] missing 'server' key, got: {scripts}"
    assert ":main" in scripts["server"], f"server entry must reference main(), got: {scripts['server']}"
    deps = [dep.lower() for dep in d.get("project", {}).get("dependencies", [])]
    has_openenv = any("openenv" in dep for dep in deps)
    assert has_openenv, f"Missing openenv dependency in pyproject.toml: {deps}"
    return f"server={scripts['server']}, openenv dep ✓"

@check("uv.lock exists", "OpenEnv-core CLI")
def _():
    assert os.path.exists("uv.lock"), "Missing uv.lock — run 'uv lock' to generate"
    size = os.path.getsize("uv.lock")
    assert size > 1000, f"uv.lock seems empty ({size} bytes)"
    return f"{size//1000}KB"

@check("server/app.py exists with def main() and if __name__", "OpenEnv-core CLI")
def _():
    path = "server/app.py"
    assert os.path.exists(path), f"Missing {path}"
    content = open(path).read()
    assert "def main(" in content, "server/app.py missing def main("
    assert "__name__" in content, "server/app.py missing if __name__ == '__main__'"
    assert "main()" in content, "server/app.py main() not called in __main__ block"
    return "def main() + __main__ block ✓"

@check("openenv validate . passes", "OpenEnv-core CLI")
def _():
    import subprocess
    result = subprocess.run(
        ["openenv", "validate", "."],
        capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"openenv validate failed:\n{output}"
    assert "[OK]" in output, f"Expected [OK] in output:\n{output}"
    return output.strip().replace("\n", " ")

@check("server/app.py runtime: GET /health returns status=healthy", "OpenEnv-core CLI")
def _():
    import sys
    sys.path.insert(0, ".")
    from fastapi.testclient import TestClient
    from server.app import app
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy", f"Expected 'healthy': {r.json()}"
    return f"status={r.json()['status']} ✓"

@check("server/app.py runtime: GET /metadata has name + description", "OpenEnv-core CLI")
def _():
    from fastapi.testclient import TestClient
    from server.app import app
    c = TestClient(app)
    r = c.get("/metadata")
    assert r.status_code == 200
    m = r.json()
    assert isinstance(m.get("name"), str)
    assert isinstance(m.get("description"), str)
    return f"name={m['name']} ✓"

@check("server/app.py runtime: GET /schema has action+observation+state", "OpenEnv-core CLI")
def _():
    from fastapi.testclient import TestClient
    from server.app import app
    c = TestClient(app)
    r = c.get("/schema")
    assert r.status_code == 200
    s = r.json()
    for key in ["action", "observation", "state"]:
        assert isinstance(s.get(key), dict), f"schema missing '{key}'"
    return f"keys={list(s.keys())} ✓"

@check("server/app.py runtime: POST /mcp returns jsonrpc=2.0", "OpenEnv-core CLI")
def _():
    from fastapi.testclient import TestClient
    from server.app import app
    c = TestClient(app)
    r = c.post("/mcp", json={})
    assert r.status_code == 200
    assert r.json().get("jsonrpc") == "2.0"
    return "jsonrpc=2.0 ✓"

@check("server/app.py runtime: all 6 openenv validate criteria pass", "OpenEnv-core CLI")
def _():
    import sys, requests as _req
    sys.path.insert(0, ".")
    from fastapi.testclient import TestClient
    from server.app import app
    from openenv.cli._validation import validate_running_environment

    c = TestClient(app)
    class MR:
        def __init__(self, r): self._r=r; self.status_code=r.status_code
        def json(self): return self._r.json()

    og, op = _req.get, _req.post
    _req.get  = lambda u,**kw: MR(c.get(u.replace("http://testserver","")))
    _req.post = lambda u,**kw: MR(c.post(u.replace("http://testserver",""), json=kw.get("json",{})))
    try:
        report = validate_running_environment("http://testserver", timeout_s=10)
    finally:
        _req.get, _req.post = og, op

    failed = [cr["id"] for cr in report["criteria"] if not cr["passed"]]
    assert report["passed"], f"Failed criteria: {failed}"
    n = report["summary"]["passed_count"]
    return f"{n}/6 criteria passed ✓"


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    ok = run_all()
    print(f"\nValidator finished in {time.time()-t0:.1f}s")
    sys.exit(0 if ok else 1)
