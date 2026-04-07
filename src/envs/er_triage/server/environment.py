"""
environment.py — ER Triage simulation engine.

Three difficulty tasks controlled by config:

  EASY   — 10 patients, 3 beds, clean vitals, no comorbidities.
  MEDIUM — 20 patients, 5 beds, comorbidities amplify risk, fast deterioration.
  HARD   — 30 patients, 8 beds, noisy vitals, surges, bed timeouts.

Reward components (all returned in ERTriageReward):
  +severity_true × 10  admitted patient saved
  +2                   bonus for choosing highest-severity patient
  -0.5 × wait/60       per-step wait penalty for whole queue
  -20                  per patient death
  -5                   holding queue when beds are free
"""

import random
import uuid
from typing import List, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from envs.er_triage.models import (
    ERTriageAction, ERTriageObservation, ERTriageReward,
    ERTriageState, PatientVitals, StepResult,
)

TASK_CONFIG = {
    "easy": {
        "queue_size": 10, "beds": 3, "max_steps": 20,
        "comorbidities": False, "noisy_vitals": False,
        "surge_events": False, "bed_timeout": False,
        "noise_std": 0.0, "deterioration_range": (0.02, 0.06),
    },
    "medium": {
        "queue_size": 20, "beds": 5, "max_steps": 40,
        "comorbidities": True, "noisy_vitals": False,
        "surge_events": False, "bed_timeout": False,
        "noise_std": 0.0, "deterioration_range": (0.02, 0.15),
    },
    "hard": {
        "queue_size": 30, "beds": 8, "max_steps": 60,
        "comorbidities": True, "noisy_vitals": True,
        "surge_events": True, "bed_timeout": True,
        "noise_std": 0.12, "deterioration_range": (0.02, 0.20),
    },
}


def _compute_severity(p: PatientVitals, use_comorbidities: bool) -> float:
    """NEWS2-inspired severity score, 0.0 (healthy) → 1.0 (critical)."""
    score = 0.0
    score += min(abs(p.heart_rate - 70) / 70 * 0.20, 0.20)
    if p.spo2 < 95:
        score += (95 - p.spo2) / 10 * 0.30
    if p.bp_systolic < 90:
        score += (90 - p.bp_systolic) / 90 * 0.25
    elif p.bp_systolic > 180:
        score += (p.bp_systolic - 180) / 100 * 0.15
    if p.respiratory_rate < 10 or p.respiratory_rate > 25:
        score += 0.10
    if p.temperature > 38.5:
        score += min((p.temperature - 38.5) / 3.0 * 0.10, 0.10)
    score += p.pain_score / 10 * 0.10
    if p.age > 65:
        score += min((p.age - 65) / 100, 0.10)
    if use_comorbidities and p.comorbidities > 0:
        score *= (1 + p.comorbidities * 0.15)
    return min(score, 1.0)


def _generate_patient(pid: int, cfg: dict, rng: random.Random) -> PatientVitals:
    det_lo, det_hi = cfg["deterioration_range"]
    band = rng.choices(["mild", "moderate", "severe"], weights=[0.5, 0.35, 0.15])[0]

    if band == "mild":
        hr, bp, spo2, rr, temp, pain = rng.gauss(75,8), rng.gauss(120,10), rng.gauss(98,1), rng.gauss(16,2), rng.gauss(36.8,0.3), rng.uniform(0,3)
    elif band == "moderate":
        hr, bp, spo2, rr, temp, pain = rng.gauss(95,12), rng.gauss(105,15), rng.gauss(94,2), rng.gauss(22,3), rng.gauss(38.0,0.5), rng.uniform(3,7)
    else:
        hr, bp, spo2, rr, temp, pain = rng.gauss(120,20), rng.gauss(85,15), rng.gauss(88,4), rng.gauss(28,4), rng.gauss(39.2,0.7), rng.uniform(7,10)

    age = rng.gauss(52, 22)
    comorbidities = rng.randint(0, 3) if cfg["comorbidities"] else 0

    p = PatientVitals(
        patient_id=pid,
        heart_rate=max(30.0, hr),
        bp_systolic=max(60.0, bp),
        spo2=min(100.0, max(70.0, spo2)),
        respiratory_rate=max(6.0, rr),
        temperature=max(35.0, min(42.0, temp)),
        pain_score=max(0.0, min(10.0, pain)),
        age=max(1.0, min(100.0, age)),
        wait_minutes=rng.uniform(0, 30),
        deterioration_rate=rng.uniform(det_lo, det_hi),
        comorbidities=comorbidities,
    )

    sev = _compute_severity(p, cfg["comorbidities"])

    if cfg["noisy_vitals"] and cfg["noise_std"] > 0:
        std = cfg["noise_std"]
        noisy_p = p.model_copy(update={
            "heart_rate": p.heart_rate + rng.gauss(0, p.heart_rate * std),
            "bp_systolic": p.bp_systolic + rng.gauss(0, p.bp_systolic * std),
            "spo2": min(100.0, p.spo2 + rng.gauss(0, 2)),
            "respiratory_rate": p.respiratory_rate + rng.gauss(0, p.respiratory_rate * std),
            "temperature": p.temperature + rng.gauss(0, 0.3),
            "pain_score": max(0.0, min(10.0, p.pain_score + rng.gauss(0, 1))),
        })
        sev_noisy = min(1.0, max(0.0, _compute_severity(noisy_p, cfg["comorbidities"])))
    else:
        sev_noisy = sev

    return p.model_copy(update={
        "severity_true": round(sev, 4),
        "severity_noisy": round(sev_noisy, 4),
    })


# Internal mutable patient record (not exposed via API directly)
class _Patient:
    __slots__ = ["vitals", "severity_true", "severity_noisy"]
    def __init__(self, vitals: PatientVitals):
        self.vitals = vitals
        self.severity_true  = vitals.severity_true or 0.0
        self.severity_noisy = vitals.severity_noisy


class ERTriageEnvironment:
    def __init__(self, task: str = "easy", seed: Optional[int] = None):
        if task not in TASK_CONFIG:
            raise ValueError(f"task must be one of {list(TASK_CONFIG)}")
        self.task   = task
        self.seed   = seed
        self._cfg   = TASK_CONFIG[task]
        self._rng   = random.Random(seed)
        self._state: Optional[ERTriageState] = None
        self._queue: List[PatientVitals] = []
        self._beds_occupied: int = 0
        self._bed_timers: List[int] = []
        self._pid_counter: int = 0

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self) -> ERTriageObservation:
        self._rng = random.Random(self._seed_for_episode())
        self._beds_occupied = 0
        self._bed_timers = []
        self._pid_counter = 0

        self._queue = [
            _generate_patient(i, self._cfg, self._rng)
            for i in range(self._cfg["queue_size"])
        ]
        self._pid_counter = self._cfg["queue_size"]

        self._state = ERTriageState(
            episode_id=str(uuid.uuid4())[:8],
            step_count=0, task=self.task,
            admitted_count=0, deaths=0, total_reward=0.0,
            queue_size=len(self._queue),
            beds_total=self._cfg["beds"],
            beds_available=self._cfg["beds"],
        )
        return self._make_obs(reward=None)

    def _seed_for_episode(self) -> int:
        return self.seed if self.seed is not None else self._rng.randint(0, 2**31)

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: ERTriageAction) -> StepResult:
        assert self._state is not None, "Call reset() before step()"
        cfg = self._cfg
        s   = self._state
        s.step_count += 1

        # Breakdown accumulators
        severity_saved = 0.0
        priority_bonus = 0.0
        wait_penalty   = 0.0
        death_penalty  = 0.0
        hold_penalty   = 0.0

        # Free timed-out beds
        if cfg["bed_timeout"]:
            self._bed_timers = [t - 1 for t in self._bed_timers]
            freed = self._bed_timers.count(0)
            self._beds_occupied = max(0, self._beds_occupied - freed)
            self._bed_timers = [t for t in self._bed_timers if t > 0]

        beds_free = cfg["beds"] - self._beds_occupied

        # Deteriorate waiting patients
        dead = []
        for i, p in enumerate(self._queue):
            updated = p.model_copy(update={
                "wait_minutes": p.wait_minutes + 1,
                "severity_true": min(1.0, (p.severity_true or 0.0) + p.deterioration_rate),
                "severity_noisy": min(1.0, max(0.0,
                    p.severity_noisy + p.deterioration_rate
                    + (self._rng.gauss(0, cfg["noise_std"] * 0.05) if cfg["noisy_vitals"] else 0)
                )),
            })
            self._queue[i] = updated
            if updated.severity_true >= 1.0:
                dead.append(i)
                s.deaths += 1
                death_penalty -= 20.0

        for i in sorted(dead, reverse=True):
            self._queue.pop(i)

        # Wait penalty
        total_wait = sum(p.wait_minutes for p in self._queue)
        wait_penalty = -0.5 * total_wait / 60.0

        # Process action
        if action.patient_index == -1:
            if beds_free > 0 and self._queue:
                hold_penalty = -5.0
        elif 0 <= action.patient_index < len(self._queue) and beds_free > 0:
            patient = self._queue.pop(action.patient_index)
            sev = patient.severity_true or 0.0
            severity_saved = sev * 10.0
            if self._queue:
                max_sev = max(p.severity_true or 0.0 for p in self._queue)
                if sev >= max_sev * 0.90:
                    priority_bonus = 2.0
            self._beds_occupied += 1
            s.admitted_count += 1
            if cfg["bed_timeout"]:
                self._bed_timers.append(int(self._rng.uniform(3, 8) * (1 + sev)))

        # Surge
        if cfg["surge_events"] and s.step_count % 10 == 0:
            for _ in range(self._rng.randint(2, 5)):
                self._queue.append(_generate_patient(self._pid_counter, cfg, self._rng))
                self._pid_counter += 1

        total = severity_saved + priority_bonus + wait_penalty + death_penalty + hold_penalty
        reward = ERTriageReward(
            total=round(total, 4),
            severity_saved=round(severity_saved, 4),
            priority_bonus=round(priority_bonus, 4),
            wait_penalty=round(wait_penalty, 4),
            death_penalty=round(death_penalty, 4),
            hold_penalty=round(hold_penalty, 4),
        )

        s.total_reward += total
        s.queue_size    = len(self._queue)
        s.beds_available = cfg["beds"] - self._beds_occupied

        done = (
            s.step_count >= cfg["max_steps"]
            or (not self._queue and s.beds_available == cfg["beds"])
        )
        obs = self._make_obs(reward=reward, done=done)

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={"admitted": s.admitted_count, "deaths": s.deaths,
                  "step": s.step_count, "queue_size": s.queue_size},
        )

    # ── state ─────────────────────────────────────────────────────────────────
    @property
    def state(self) -> Optional[ERTriageState]:
        return self._state

    # ── helpers ───────────────────────────────────────────────────────────────
    def _make_obs(self, reward: Optional[ERTriageReward], done: bool = False) -> ERTriageObservation:
        beds_free = self._cfg["beds"] - self._beds_occupied
        # Hide severity_true in hard mode
        hide_true = self.task == "hard"
        patients = [
            p.model_copy(update={"severity_true": None}) if hide_true else p
            for p in self._queue
        ]
        return ERTriageObservation(
            patients=patients,
            beds_available=beds_free,
            step=self._state.step_count if self._state else 0,
            task=self.task,
            done=done,
            reward=reward,
            info={"admitted": self._state.admitted_count if self._state else 0,
                  "deaths": self._state.deaths if self._state else 0},
        )
