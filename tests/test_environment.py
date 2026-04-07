"""
tests/test_environment.py — Full test suite for ERTriageEnv.

Run:  PYTHONPATH=src pytest tests/ -v
"""

import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pydantic
from fastapi.testclient import TestClient

from envs.er_triage.server.environment import ERTriageEnvironment, TASK_CONFIG, _compute_severity
from envs.er_triage.models import (
    ERTriageAction, ERTriageObservation, ERTriageReward,
    ERTriageState, PatientVitals, StepResult,
)
from envs.er_triage.graders.graders import grade, greedy_agent, random_agent, fifo_agent
from envs.er_triage.server.app import app


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env_easy():
    return ERTriageEnvironment(task="easy", seed=42)

@pytest.fixture
def env_medium():
    return ERTriageEnvironment(task="medium", seed=42)

@pytest.fixture
def env_hard():
    return ERTriageEnvironment(task="hard", seed=42)

@pytest.fixture
def http():
    return TestClient(app)


# ── Pydantic models ───────────────────────────────────────────────────────────

class TestPydanticModels:
    def test_all_are_pydantic(self):
        for cls in [ERTriageAction, ERTriageObservation, ERTriageReward,
                    ERTriageState, PatientVitals, StepResult]:
            assert issubclass(cls, pydantic.BaseModel), f"{cls.__name__} not Pydantic"

    def test_action_patient_index(self):
        a = ERTriageAction(patient_index=5)
        assert a.patient_index == 5

    def test_action_hold(self):
        a = ERTriageAction(patient_index=-1)
        assert a.patient_index == -1

    def test_reward_has_six_fields(self):
        r = ERTriageReward(total=1.0, severity_saved=2.0, priority_bonus=0.0,
                           wait_penalty=-1.0, death_penalty=0.0, hold_penalty=0.0)
        assert r.total == 1.0

    def test_reward_breakdown_fields_exist(self):
        for field in ["total","severity_saved","priority_bonus",
                      "wait_penalty","death_penalty","hold_penalty"]:
            assert field in ERTriageReward.model_fields

    def test_observation_reward_is_reward_model(self):
        ann = ERTriageObservation.model_fields["reward"].annotation
        # Should be Optional[ERTriageReward]
        import typing
        args = typing.get_args(ann)
        assert ERTriageReward in args, "reward field must be Optional[ERTriageReward]"

    def test_step_result_fields(self):
        for field in ["observation","reward","done","info"]:
            assert field in StepResult.model_fields

    def test_patient_vitals_fields(self):
        required = {"heart_rate","bp_systolic","spo2","respiratory_rate",
                    "temperature","pain_score","age","wait_minutes",
                    "deterioration_rate","comorbidities","severity_noisy"}
        assert required.issubset(PatientVitals.model_fields.keys())

    def test_json_roundtrip(self):
        env = ERTriageEnvironment(task="easy", seed=0)
        env.reset()
        result = env.step(ERTriageAction(patient_index=0))
        j = result.model_dump_json()
        r2 = StepResult.model_validate_json(j)
        assert abs(r2.reward.total - result.reward.total) < 0.001


# ── Severity scoring ──────────────────────────────────────────────────────────

class TestSeverityScoring:
    def _patient(self, **kwargs):
        defaults = dict(patient_id=0, heart_rate=75, bp_systolic=120, spo2=98,
                        respiratory_rate=16, temperature=36.8, pain_score=2,
                        age=45, wait_minutes=5, deterioration_rate=0.03)
        defaults.update(kwargs)
        return PatientVitals(**defaults)

    def test_healthy_is_low_severity(self):
        assert _compute_severity(self._patient(), False) < 0.30

    def test_low_spo2_raises_severity(self):
        low = _compute_severity(self._patient(spo2=85), False)
        normal = _compute_severity(self._patient(spo2=98), False)
        assert low > normal

    def test_shock_bp_raises_severity(self):
        shock = _compute_severity(self._patient(bp_systolic=70), False)
        normal = _compute_severity(self._patient(bp_systolic=120), False)
        assert shock > normal

    def test_comorbidities_amplify(self):
        p0 = self._patient(spo2=93, comorbidities=0)
        p3 = self._patient(spo2=93, comorbidities=3)
        assert _compute_severity(p3, True) > _compute_severity(p0, True)

    def test_severity_bounded(self):
        p = self._patient(spo2=70, bp_systolic=60, heart_rate=180,
                          respiratory_rate=35, temperature=41.5, pain_score=10, age=95)
        s = _compute_severity(p, True)
        assert 0.0 <= s <= 1.0

    def test_spo2_monotone(self):
        sevs = [_compute_severity(self._patient(spo2=v), False)
                for v in [100, 97, 94, 91, 88]]
        assert sevs == sorted(sevs), "severity must increase as SpO2 falls"


# ── Environment API ───────────────────────────────────────────────────────────

class TestEnvironmentAPI:
    def test_reset_returns_observation(self, env_easy):
        obs = env_easy.reset()
        assert isinstance(obs, ERTriageObservation)
        assert obs.step == 0
        assert obs.reward is None
        assert obs.done is False
        assert len(obs.patients) == 10

    def test_state_none_before_reset(self):
        assert ERTriageEnvironment(task="easy").state is None

    def test_state_after_reset(self, env_easy):
        env_easy.reset()
        s = env_easy.state
        assert isinstance(s, ERTriageState)
        assert s.step_count == 0
        assert s.episode_id is not None

    def test_step_returns_step_result(self, env_easy):
        env_easy.reset()
        result = env_easy.step(ERTriageAction(patient_index=0))
        assert isinstance(result, StepResult)
        assert isinstance(result.reward, ERTriageReward)
        assert result.observation.step == 1
        assert result.done is False or result.done is True

    def test_reward_breakdown_sums_to_total(self, env_easy):
        env_easy.reset()
        for _ in range(5):
            result = env_easy.step(ERTriageAction(patient_index=0))
            r = result.reward
            expected = r.severity_saved + r.priority_bonus + r.wait_penalty + r.death_penalty + r.hold_penalty
            assert abs(r.total - expected) < 0.01
            if result.done:
                break

    def test_admission_reduces_queue(self, env_easy):
        obs = env_easy.reset()
        n = len(obs.patients)
        result = env_easy.step(ERTriageAction(patient_index=0))
        assert len(result.observation.patients) == n - 1

    def test_admission_reduces_beds(self, env_easy):
        obs = env_easy.reset()
        beds = obs.beds_available
        result = env_easy.step(ERTriageAction(patient_index=0))
        assert result.observation.beds_available == beds - 1

    def test_hold_penalty_when_beds_free(self, env_easy):
        env_easy.reset()
        result = env_easy.step(ERTriageAction(patient_index=-1))
        assert result.reward.hold_penalty < 0

    def test_episode_terminates(self):
        env = ERTriageEnvironment(task="easy", seed=1)
        env.reset()
        steps = 0
        done = False
        while not done and steps < 25:
            r = env.step(ERTriageAction(patient_index=-1))
            done = r.done
            steps += 1
        assert done
        assert env.state.step_count <= TASK_CONFIG["easy"]["max_steps"]

    def test_reproducible_with_seed(self):
        e1, e2 = ERTriageEnvironment(task="hard", seed=99), ERTriageEnvironment(task="hard", seed=99)
        o1, o2 = e1.reset(), e2.reset()
        for p1, p2 in zip(o1.patients, o2.patients):
            assert abs(p1.heart_rate - p2.heart_rate) < 0.01

    def test_all_tasks_run(self):
        for task in ["easy", "medium", "hard"]:
            env = ERTriageEnvironment(task=task, seed=0)
            obs = env.reset()
            assert not obs.done
            r = env.step(ERTriageAction(patient_index=0))
            assert r.observation.step == 1

    def test_invalid_task(self):
        with pytest.raises(ValueError):
            ERTriageEnvironment(task="impossible")

    def test_step_before_reset(self):
        with pytest.raises(AssertionError):
            ERTriageEnvironment(task="easy").step(ERTriageAction(patient_index=0))

    def test_hard_hides_severity_true(self, env_hard):
        # severity_true should be None in hard mode (hidden from agent)
        obs = env_hard.reset()
        for p in obs.patients:
            assert p.severity_true is None

    def test_easy_exposes_severity_true(self, env_easy):
        obs = env_easy.reset()
        truths = [p.severity_true for p in obs.patients if p.severity_true is not None]
        assert len(truths) > 0

    def test_deterioration_worsens_severity(self):
        env = ERTriageEnvironment(task="easy", seed=5)
        obs = env.reset()
        init_max = max(p.severity_noisy for p in obs.patients)
        for _ in range(3):
            r = env.step(ERTriageAction(patient_index=-1))
            if r.done: break
        if r.observation.patients:
            final_max = max(p.severity_noisy for p in r.observation.patients)
            assert final_max >= init_max * 0.95  # at least as bad

    def test_hard_noisy_vitals(self, env_hard):
        obs = env_hard.reset()
        # In hard mode, severity_noisy != severity_true (they differ due to noise)
        # We can't compare directly (true is hidden), but check severity_noisy is in [0,1]
        for p in obs.patients:
            assert 0.0 <= p.severity_noisy <= 1.0

    def test_medium_has_comorbidities(self, env_medium):
        obs = env_medium.reset()
        assert any(p.comorbidities > 0 for p in obs.patients)

    def test_reward_non_sparse(self):
        env = ERTriageEnvironment(task="medium", seed=7)
        env.reset()
        rewards = []
        for i in range(8):
            r = env.step(ERTriageAction(patient_index=i % 5))
            rewards.append(round(r.reward.total, 1))
            if r.done: break
        assert len(set(rewards)) >= 2


# ── Graders ───────────────────────────────────────────────────────────────────

class TestGraders:
    @pytest.mark.parametrize("task", ["easy","medium","hard"])
    def test_score_in_range(self, task):
        r = grade(task, greedy_agent)
        assert 0.0 <= r["score"] <= 1.0

    @pytest.mark.parametrize("task", ["easy","medium","hard"])
    def test_deterministic(self, task):
        s1 = grade(task, greedy_agent)["score"]
        s2 = grade(task, greedy_agent)["score"]
        assert s1 == s2

    def test_greedy_beats_random(self):
        for task in ["easy","medium","hard"]:
            assert grade(task, greedy_agent)["score"] >= grade(task, random_agent)["score"]

    def test_score_decomposition(self):
        r = grade("medium", greedy_agent)
        exp = round(0.50*r["deaths_score"] + 0.30*r["priority_score"] + 0.20*r["efficiency_score"], 4)
        assert abs(r["score"] - exp) < 0.001

    def test_required_fields(self):
        r = grade("easy", greedy_agent)
        for f in ["score","deaths_score","priority_score","efficiency_score",
                  "avg_deaths","avg_priority_rate","avg_hold_rate","episodes","task"]:
            assert f in r

    def test_10_episodes(self):
        assert grade("easy", greedy_agent)["episodes"] == 10


# ── HTTP API ──────────────────────────────────────────────────────────────────

class TestHTTPAPI:
    def test_health(self, http):
        r = http.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_tasks(self, http):
        r = http.get("/tasks")
        assert r.status_code == 200
        names = {t["name"] for t in r.json()}
        assert names == {"easy","medium","hard"}

    def test_reset_returns_observation(self, http):
        r = http.post("/reset", json={"task":"easy","seed":42})
        assert r.status_code == 200
        obs = r.json()
        assert obs["step"] == 0
        assert obs["reward"] is None
        assert obs["done"] is False
        assert len(obs["patients"]) == 10

    def test_step_returns_step_result(self, http):
        http.post("/reset", json={"task":"easy","seed":0})
        r = http.post("/step", json={"patient_index":0})
        assert r.status_code == 200
        result = r.json()
        # Must have StepResult shape: observation, reward, done, info
        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert "info" in result
        # reward must be typed ERTriageReward
        reward = result["reward"]
        for field in ["total","severity_saved","priority_bonus","wait_penalty",
                      "death_penalty","hold_penalty"]:
            assert field in reward, f"reward missing field: {field}"

    def test_state_after_step(self, http):
        http.post("/reset", json={"task":"easy","seed":0})
        http.post("/step", json={"patient_index":0})
        r = http.get("/state")
        assert r.status_code == 200
        assert r.json()["admitted_count"] == 1
        assert r.json()["step_count"] == 1

    def test_grade_score_in_range(self, http):
        for task in ["easy","medium","hard"]:
            r = http.post("/grade", json={"task":task,"actions":[0,1,0,2,1]})
            assert r.status_code == 200
            assert 0.0 <= r.json()["score"] <= 1.0

    def test_reset_invalid_task(self, http):
        r = http.post("/reset", json={"task":"impossible"})
        assert r.status_code == 400

    def test_hard_hides_severity_true(self, http):
        r = http.post("/reset", json={"task":"hard","seed":1})
        patients = r.json()["patients"]
        assert all(p["severity_true"] is None for p in patients)

    def test_easy_exposes_severity_true(self, http):
        r = http.post("/reset", json={"task":"easy","seed":1})
        truths = [p["severity_true"] for p in r.json()["patients"] if p["severity_true"] is not None]
        assert len(truths) > 0

    def test_full_episode_completes(self, http):
        http.post("/reset", json={"task":"easy","seed":99})
        steps, done = 0, False
        while not done and steps < 25:
            r = http.post("/step", json={"patient_index":0})
            done = r.json()["done"]
            steps += 1
        assert done

    def test_reward_breakdown_sums_via_http(self, http):
        http.post("/reset", json={"task":"medium","seed":3})
        for _ in range(5):
            r = http.post("/step", json={"patient_index":0})
            reward = r.json()["reward"]
            parts = (reward["severity_saved"] + reward["priority_bonus"] +
                     reward["wait_penalty"] + reward["death_penalty"] + reward["hold_penalty"])
            assert abs(reward["total"] - parts) < 0.01
            if r.json()["done"]: break


# ── Spec compliance ───────────────────────────────────────────────────────────

class TestSpecCompliance:
    def test_openenv_yaml_exists(self):
        assert os.path.exists(os.path.join(os.path.dirname(__file__),"..","openenv.yaml"))

    def test_openenv_yaml_pydantic_models(self):
        import yaml
        with open(os.path.join(os.path.dirname(__file__),"..","openenv.yaml")) as f:
            spec = yaml.safe_load(f)
        models = spec.get("models", {})
        for key in ["action","observation","reward"]:
            assert key in models, f"models.{key} missing from openenv.yaml"
            assert "pydantic" in models[key].get("type","").lower()

    def test_openenv_yaml_openai_api_key(self):
        import yaml
        with open(os.path.join(os.path.dirname(__file__),"..","openenv.yaml")) as f:
            spec = yaml.safe_load(f)
        creds = spec.get("inference",{}).get("credentials",[])
        assert "OPENAI_API_KEY" in creds

    def test_inference_uses_openai_api_key(self):
        with open(os.path.join(os.path.dirname(__file__),"..","inference.py")) as f:
            src = f.read()
        assert "OPENAI_API_KEY" in src
        assert "from openai import OpenAI" in src

    def test_dockerfile_hf_ready(self):
        with open(os.path.join(os.path.dirname(__file__),"..","Dockerfile")) as f:
            d = f.read()
        assert "7860" in d
        assert "appuser" in d

    def test_readme_has_openenv_tag(self):
        with open(os.path.join(os.path.dirname(__file__),"..","README.md")) as f:
            content = f.read()
        assert "openenv" in content
        assert "sdk: docker" in content

    def test_difficulty_progression(self):
        easy, medium, hard = TASK_CONFIG["easy"], TASK_CONFIG["medium"], TASK_CONFIG["hard"]
        assert easy["queue_size"] < medium["queue_size"] < hard["queue_size"]
        assert easy["max_steps"]  < medium["max_steps"]  < hard["max_steps"]
        assert easy["beds"]       < medium["beds"]        < hard["beds"]


# ── OpenEnv runtime criteria (server/app.py) ──────────────────────────────────

class TestOpenEnvRuntime:
    """
    Tests every criterion that `openenv validate <url>` checks.
    Uses server/app.py (the spec-compliant entry point) not the legacy src/ app.
    """

    @pytest.fixture(autouse=True)
    def server_client(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from server.app import app as server_app
        self.c = TestClient(server_app)

    def test_openapi_has_version(self):
        r = self.c.get("/openapi.json")
        assert r.status_code == 200
        version = r.json().get("info", {}).get("version")
        assert isinstance(version, str) and version, f"info.version missing: {r.json()}"

    def test_health_returns_healthy(self):
        r = self.c.get("/health")
        assert r.status_code == 200
        assert r.json().get("status") == "healthy", f"Expected 'healthy', got: {r.json()}"

    def test_metadata_name_and_description(self):
        r = self.c.get("/metadata")
        assert r.status_code == 200
        m = r.json()
        assert isinstance(m.get("name"), str) and m["name"]
        assert isinstance(m.get("description"), str) and m["description"]

    def test_schema_has_action_observation_state(self):
        r = self.c.get("/schema")
        assert r.status_code == 200
        s = r.json()
        assert isinstance(s.get("action"), dict),      "schema missing action"
        assert isinstance(s.get("observation"), dict), "schema missing observation"
        assert isinstance(s.get("state"), dict),       "schema missing state"

    def test_mcp_returns_jsonrpc_2(self):
        r = self.c.post("/mcp", json={})
        assert r.status_code == 200
        assert r.json().get("jsonrpc") == "2.0", f"Expected jsonrpc=2.0: {r.json()}"

    def test_openapi_paths_include_reset_step_state(self):
        r = self.c.get("/openapi.json")
        paths = r.json().get("paths", {})
        for p in ["/reset", "/step", "/state"]:
            assert p in paths, f"OpenAPI paths missing {p}"

    def test_reset_empty_body_returns_200(self):
        """Validation script sends POST /reset with {} body."""
        r = self.c.post("/reset", json={})
        assert r.status_code == 200
        obs = r.json()
        assert "patients" in obs
        assert len(obs["patients"]) > 0

    def test_reset_with_task_and_seed(self):
        r = self.c.post("/reset", json={"task": "medium", "seed": 42})
        assert r.status_code == 200
        obs = r.json()
        assert obs["task"] == "medium"
        assert len(obs["patients"]) == 20

    def test_step_returns_step_result_shape(self):
        self.c.post("/reset", json={"task": "easy", "seed": 0})
        r = self.c.post("/step", json={"patient_index": 0})
        assert r.status_code == 200
        result = r.json()
        assert "observation" in result, "StepResult missing observation"
        assert "reward" in result,      "StepResult missing reward"
        assert "done" in result,        "StepResult missing done"
        assert "info" in result,        "StepResult missing info"
        assert "total" in result["reward"], "reward missing total"

    def test_reward_has_full_breakdown(self):
        self.c.post("/reset", json={"task": "easy", "seed": 1})
        r = self.c.post("/step", json={"patient_index": 0})
        reward = r.json()["reward"]
        for field in ["total","severity_saved","priority_bonus",
                      "wait_penalty","death_penalty","hold_penalty"]:
            assert field in reward, f"reward missing {field}"

    def test_state_returns_episode_metadata(self):
        self.c.post("/reset", json={"task": "easy", "seed": 0})
        self.c.post("/step", json={"patient_index": 0})
        r = self.c.get("/state")
        assert r.status_code == 200
        s = r.json()
        assert "step_count" in s or "step" in s
        assert "admitted_count" in s or "admitted" in s

    def test_grade_score_in_range(self):
        for task in ["easy", "medium", "hard"]:
            r = self.c.post("/grade", json={"task": task, "actions": [0, 1, 0]})
            assert r.status_code == 200
            score = r.json()["score"]
            assert 0.0 <= score <= 1.0, f"{task} score {score} out of [0,1]"

    def test_tasks_lists_three_difficulties(self):
        r = self.c.get("/tasks")
        assert r.status_code == 200
        names = {t["name"] for t in r.json()}
        assert names == {"easy", "medium", "hard"}

    def test_hard_reset_has_thirty_patients(self):
        r = self.c.post("/reset", json={"task": "hard", "seed": 7})
        assert r.status_code == 200
        assert len(r.json()["patients"]) == 30

    def test_full_openenv_validate_runtime_criteria(self):
        """Mirrors exactly what openenv validate <url> checks."""
        from openenv.cli._validation import validate_running_environment
        import requests as _req

        c = self.c
        class MockResponse:
            def __init__(self, r):
                self._r = r
                self.status_code = r.status_code
            def json(self):
                return self._r.json()

        orig_get  = _req.get
        orig_post = _req.post
        _req.get  = lambda url, **kw: MockResponse(c.get(url.replace("http://testserver", "")))
        _req.post = lambda url, **kw: MockResponse(c.post(url.replace("http://testserver", ""), json=kw.get("json", {})))

        try:
            report = validate_running_environment("http://testserver", timeout_s=10)
        finally:
            _req.get  = orig_get
            _req.post = orig_post

        failed = [c["id"] for c in report["criteria"] if not c["passed"]]
        assert report["passed"], f"openenv validate runtime FAILED: {failed}\n{report}"
        assert report["summary"]["passed_count"] == 6, f"Expected 6/6: {report['summary']}"
