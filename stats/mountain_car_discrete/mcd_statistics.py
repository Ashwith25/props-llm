# stats/mc_statistics.py
from typing import Sequence, Dict, Any
import numpy as np
import pandas as pd
import gymnasium as gym
from stats.base_statistics import BaseStatistics


class MC_Statistics(BaseStatistics):
    """
    MountainCar-v0 statistics collector.

    Implements the BaseStatistics interface used by your evaluation harness:
      - initialise_eval(): reset per-episode buffers
      - run_eval(env, obs, u, reward): record step-wise signals
      - post_eval(env, terminated, truncated): finalize per-episode metrics
      - evaluate_params(params): aggregate ONE JSON summary across episodes

    Notes
    - Environment: MountainCar-v0 (Discrete actions: 0=left, 1=idle, 2=right)
    - Termination: x >= 0.5 (goal). Truncation: T == 200 steps (horizon).  # Gymnasium docs
    """

    def __init__(self):
        # kept for parity with your IDP class; harness can rely on these defaults
        self.EPISODES = 20
        self.HORIZON = 200
        self.GOAL_X = 0.5

    # ---------------- helpers ----------------
    def _bias_index(self, arr: np.ndarray) -> float:
        """
        Signed residency in [-1, 1]: sum(arr) / sum(|arr|).
        0 ~ balanced; + right-leaning; - left-leaning.
        """
        denom = np.sum(np.abs(arr))
        return float(np.sum(arr) / denom) if denom > 1e-12 else 0.0

    def _rms(self, arr: np.ndarray) -> float:
        return float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0

    def _p95_abs(self, arr: np.ndarray) -> float:
        return float(np.percentile(np.abs(arr), 95)) if arr.size else 0.0

    def _zero_cross_rate(self, arr: np.ndarray) -> float:
        if arr.size <= 1:
            return 0.0
        s = np.sign(arr)
        return float(np.mean(s[1:] != s[:-1]))

    def _longest_streak(self, mask: np.ndarray) -> int:
        best = cur = 0
        for v in mask:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return int(best)

    def _default_failure_label(self, terminated: bool, truncated: bool) -> str:
        if terminated and not truncated:
            return "terminated"
        if truncated and not terminated:
            return "time-limit"
        if terminated and truncated:
            return "terminated+truncated"
        return "unknown"

    def _action_dir(self, a: np.ndarray) -> np.ndarray:
        """
        Map discrete actions -> direction in {-1, 0, +1}:
          0 (left) -> -1, 1 (idle) -> 0, 2 (right) -> +1
        """
        out = np.zeros_like(a, dtype=float)
        out[a == 0] = -1.0
        out[a == 2] = +1.0
        return out

    # ---------------- BaseStatistics API ----------------
    def initialise_eval(self):
        self.t = 0
        self.ep_return = 0.0
        self.x_list: list[float] = []
        self.v_list: list[float] = []
        self.a_list: list[int] = []
        self.per_ep = []

    def run_eval(self, env, obs, u, reward):
        # obs = [position, velocity]
        o = np.asarray(obs).ravel()
        x = float(o[0]) if o.size > 0 else 0.0
        v = float(o[1]) if o.size > 1 else 0.0

        # record action as an integer in {0,1,2} (robust to stray types)
        try:
            a = int(np.asarray(u).item())
        except Exception:
            a = int(u) if isinstance(u, (int, np.integer)) else 1  # default idle

        # clamp to valid domain (defensive)
        a = int(np.clip(a, 0, 2))

        self.x_list.append(x)
        self.v_list.append(v)
        self.a_list.append(a)

        self.ep_return += float(reward)
        self.t += 1

    def post_eval(self, env, terminated, truncated):
        x_arr = np.asarray(self.x_list, dtype=float)
        v_arr = np.asarray(self.v_list, dtype=float)
        a_arr = np.asarray(self.a_list, dtype=int)
        length = int(self.t)
        ret = float(self.ep_return)
        failure_mode = self._default_failure_label(terminated, truncated)

        # --- Coverage / goal proximity ---
        right_half_frac = float(np.mean(x_arr >= 0.0)) if length else 0.0
        left_half_frac = float(np.mean(x_arr < 0.0)) if length else 0.0
        valley_frac = float(np.mean(x_arr <= -0.9)) if length else 0.0  # near left wall
        goal_zone_frac = float(np.mean(x_arr >= (self.GOAL_X - 0.05))) if length else 0.0

        x_max = float(np.max(x_arr)) if length else -np.inf
        x_min = float(np.min(x_arr)) if length else +np.inf
        x_range = float(x_max - x_min) if length else 0.0

        # --- Bias / smoothness ---
        drift_x_index = self._bias_index(x_arr)
        rms_x = self._rms(x_arr)
        rms_v = self._rms(v_arr)
        v_p95 = self._p95_abs(v_arr)
        zcr_v = self._zero_cross_rate(v_arr)  # turning points rate
        stall_rate = float(np.mean(np.abs(v_arr) < 1e-3)) if length else 0.0

        # --- Actions (discrete) ---
        left_rate = float(np.mean(a_arr == 0)) if length else 0.0
        idle_rate = float(np.mean(a_arr == 1)) if length else 0.0
        right_rate = float(np.mean(a_arr == 2)) if length else 0.0
        action_flip_rate = float(np.mean(a_arr[1:] != a_arr[:-1])) if length > 1 else 0.0

        a_dir = self._action_dir(a_arr)              # {-1,0,+1}
        mean_push_rate = float(np.mean(a_dir != 0))  # energy proxy
        smoothness_a = float(np.mean(np.abs(np.diff(a_dir)))) if length > 1 else 0.0
        push_bias_index = self._bias_index(a_dir)

        # action-velocity coordination (sign alignment)
        try:
            corr_action_vel = float(np.corrcoef(a_dir, v_arr)[0, 1]) if length > 1 else 0.0
        except Exception:
            corr_action_vel = 0.0

        # --- Goal timing / bounces ---
        success = bool(terminated and not truncated)
        if length:
            goal_hits = np.where(x_arr >= self.GOAL_X)[0]
            steps_to_goal = int(goal_hits[0] + 1) if goal_hits.size else np.nan
        else:
            steps_to_goal = np.nan

        # crude left-wall contact proxy (stuck at boundary)
        bounce_left = int(np.sum((x_arr <= -1.2 + 1e-9) & (np.abs(v_arr) < 1e-6)))

        self.per_ep.append({
            # outcomes
            "length": length,
            "return": ret,
            "failure_mode": failure_mode,
            "success": 1 if success else 0,
            "steps_to_goal": steps_to_goal,

            # coverage & peaks
            "x_max": x_max, "x_min": x_min, "x_range": x_range,
            "right_half_frac": right_half_frac,
            "left_half_frac": left_half_frac,
            "valley_frac": valley_frac,
            "goal_zone_frac": goal_zone_frac,

            # bias / dynamics
            "drift_x_index": drift_x_index,   # [-1,1]
            "rms_x": rms_x,
            "rms_v": rms_v,
            "v_p95": v_p95,
            "zero_cross_rate_v": zcr_v,
            "stall_rate": stall_rate,

            # actions
            "left_rate": left_rate,
            "idle_rate": idle_rate,
            "right_rate": right_rate,
            "action_flip_rate": action_flip_rate,
            "mean_push_rate": mean_push_rate,
            "smoothness_a": smoothness_a,
            "push_bias_index": push_bias_index,   # [-1,1]
            "corr_action_vel": corr_action_vel,

            # misc
            "bounce_left": bounce_left,
        })

    def evaluate_params(self, params: Sequence[float]) -> Dict[str, Any]:
        """
        Tool: Summarize ONE MountainCar-v0 iteration (all episodes) into a compact JSON.

        Input
        params (Sequence[float]): kept for interface parity; not used here.

        Output (JSON)
        {
          "meta": {"env": "MountainCar-v0", "trials": <int>, "horizon": 200, "goal_x": 0.5},
          "failures": {
            "time_limit": <int>, "terminated": <int>, "terminated_truncated": <int>, "unknown": <int>,
            "time_limit_rate": <float>, "success_rate": <float>
          },
          "stats": { "<metric>": {"median": <float>, "q1": <float>, "q3": <float>}, ... },
          "return_mean": <float>
        }

        Metrics (concise meanings)
          length, return                      : episode length and sum of rewards (higher is better)
          success, steps_to_goal              : success flag and first step reaching x >= goal (lower is better)
          x_max/x_min/x_range                 : coverage and momentum amplitude
          right_half_frac / valley_frac       : residency near goal side / near left wall
          goal_zone_frac                      : fraction of steps near the goal ridge
          drift_x_index                       : signed position bias in [-1,1] (right +, left -)
          rms_x, rms_v, v_p95, zero_cross_rate_v, stall_rate : stability/momentum proxies
          left_rate/idle_rate/right_rate      : action distribution
          action_flip_rate, smoothness_a      : control consistency (lower is smoother)
          mean_push_rate, push_bias_index     : effort / left-right bias in actions
          corr_action_vel                     : alignment of push direction with velocity
          bounce_left                         : boundary contacts (lower is better)
        """
        df = pd.DataFrame(self.per_ep)
        if len(df) == 0:
            return {
                "meta": {"env": "MountainCar-v0", "trials": 0, "horizon": self.HORIZON, "goal_x": self.GOAL_X},
                "failures": {"time_limit": 0, "terminated": 0, "terminated_truncated": 0, "unknown": 0,
                             "time_limit_rate": 0.0, "success_rate": 0.0},
                "stats": {},
                "return_mean": float("nan"),
            }

        # failure counts + success rate
        fail_counts = df["failure_mode"].value_counts()
        failures = {
            "time_limit": int(fail_counts.get("time-limit", 0)),
            "terminated": int(fail_counts.get("terminated", 0)),
            "terminated_truncated": int(fail_counts.get("terminated+truncated", 0)),
            "unknown": int(fail_counts.get("unknown", 0)),
        }
        failures["time_limit_rate"] = float(failures["time_limit"] / len(df))
        failures["success_rate"] = float(df["success"].mean())

        # summarize each numeric column (except failure_mode)
        stats: Dict[str, Dict[str, float]] = {}
        for c in df.columns:
            if c == "failure_mode":
                continue
            arr = df[c].to_numpy(dtype=float)
            q1, med, q3 = np.percentile(arr, [25, 50, 75]) if arr.size else (np.nan, np.nan, np.nan)
            stats[c] = {"median": float(med), "q1": float(q1), "q3": float(q3)}

        return {
            "meta": {"env": "MountainCar-v0", "trials": len(df), "horizon": self.HORIZON, "goal_x": self.GOAL_X},
            "failures": failures,
            "stats": stats,
            "return_mean": float(df["return"].mean()),
        }
