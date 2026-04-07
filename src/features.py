"""
features.py — Convert ERTriageObservation into fixed-size tensors for DQN.

The core challenge: the patient queue is variable length (grows/shrinks each
step), but neural networks need fixed-size inputs. We solve this by:

  1. Padding the queue to MAX_QUEUE_SIZE rows (zero-pad absent patients)
  2. Sorting patients by severity_noisy DESC so the agent always sees the
     most critical patients at the top — makes learning faster
  3. Appending global context (beds_available, step_fraction) as extra features
  4. Normalising every feature to [0, 1]

Output tensor shape: (MAX_QUEUE_SIZE * PATIENT_FEATURES + GLOBAL_FEATURES,)

For action selection:
  - Valid actions are [0 .. len(patients)-1] + [-1 (hold)]
  - We mask invalid actions (padded slots) with -inf before softmax/argmax
"""

import numpy as np
from typing import List, Tuple

from envs.er_triage.models import ERTriageObservation, PatientVitals

# Per-task max queue sizes (must cover surges in hard mode)
MAX_QUEUE = {
    "easy":   12,
    "medium": 25,
    "hard":   50,
}

# 9 features per patient slot (all normalised 0-1)
PATIENT_FEATURES = 9

# 3 global features appended at the end
GLOBAL_FEATURES = 3

# Total obs dim per task
OBS_DIM = {task: MAX_QUEUE[task] * PATIENT_FEATURES + GLOBAL_FEATURES
           for task in MAX_QUEUE}

# Action space size per task: MAX_QUEUE slots + 1 hold action
ACTION_DIM = {task: MAX_QUEUE[task] + 1 for task in MAX_QUEUE}

# Index of the "hold" action
HOLD_ACTION = {task: MAX_QUEUE[task] for task in MAX_QUEUE}


def _normalise_patient(p: PatientVitals) -> np.ndarray:
    """
    Returns 9-float array, each value in [0, 1].
    Features ordered by clinical importance so early gradient signal is rich.
    """
    return np.array([
        p.severity_noisy,                           # 0: pre-computed risk score
        max(0.0, (100 - p.spo2) / 30),             # 1: hypoxia (higher = worse)
        max(0.0, (90  - p.bp_systolic) / 90),      # 2: hypotension
        min(1.0, max(0.0, (p.heart_rate - 60) / 140)),   # 3: tachycardia
        min(1.0, max(0.0, (p.respiratory_rate - 12) / 24)),  # 4: tachypnoea
        min(1.0, max(0.0, (p.temperature - 36.0) / 6.0)),   # 5: fever
        p.pain_score / 10.0,                        # 6: pain
        min(1.0, p.wait_minutes / 120.0),           # 7: urgency from wait time
        p.deterioration_rate / 0.25,               # 8: how fast they're crashing
    ], dtype=np.float32)


def obs_to_tensor(obs: ERTriageObservation) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert observation to (state_vector, action_mask).

    state_vector : shape (OBS_DIM[task],)  — input to the network
    action_mask  : shape (ACTION_DIM[task],) — 1.0 = valid, 0.0 = masked out
    """
    task = obs.task
    max_q = MAX_QUEUE[task]
    act_dim = ACTION_DIM[task]

    # Sort patients by severity_noisy descending so consistent ordering
    sorted_patients = sorted(obs.patients, key=lambda p: p.severity_noisy, reverse=True)

    # Build patient feature matrix (max_q x PATIENT_FEATURES)
    patient_matrix = np.zeros((max_q, PATIENT_FEATURES), dtype=np.float32)
    for i, p in enumerate(sorted_patients[:max_q]):
        patient_matrix[i] = _normalise_patient(p)

    # Global context
    beds_frac  = obs.beds_available / 10.0          # normalised (max reasonable beds)
    step_frac  = obs.step / 60.0                    # normalised (max steps = 60)
    queue_frac = len(obs.patients) / max_q

    global_ctx = np.array([beds_frac, step_frac, queue_frac], dtype=np.float32)

    state_vector = np.concatenate([patient_matrix.flatten(), global_ctx])

    # Action mask: valid = patient exists in that slot OR hold action
    action_mask = np.zeros(act_dim, dtype=np.float32)
    n_patients = min(len(obs.patients), max_q)
    action_mask[:n_patients] = 1.0         # admit actions
    action_mask[HOLD_ACTION[task]] = 1.0   # hold always valid

    # Disable hold when beds are free AND patients are waiting (encourage action)
    if obs.beds_available > 0 and obs.patients:
        action_mask[HOLD_ACTION[task]] = 0.5  # still valid but discouraged

    return state_vector, action_mask


def action_index_to_env_action(action_idx: int, obs: ERTriageObservation) -> int:
    """
    Convert network action index back to patient_index for ERTriageAction.
    Handles the sort mapping: action index i → sorted_patients[i].
    Returns -1 for hold action.
    """
    task = obs.task
    if action_idx == HOLD_ACTION[task]:
        return -1
    # Recover sorted order
    sorted_patients = sorted(obs.patients, key=lambda p: p.severity_noisy, reverse=True)
    if action_idx < len(sorted_patients):
        target_id = sorted_patients[action_idx].patient_id
        # Find original index in obs.patients
        for orig_idx, p in enumerate(obs.patients):
            if p.patient_id == target_id:
                return orig_idx
    return -1  # fallback: hold
