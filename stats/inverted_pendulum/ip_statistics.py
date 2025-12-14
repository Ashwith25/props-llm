from typing import Sequence, Dict, Any
import numpy as np
import pandas as pd
from stats.base_statistics import BaseStatistics


class IP_Statistics(BaseStatistics):
    """
    InvertedPendulum-v5 statistics collector.

    Implements your BaseStatistics interface:
      - initialise_eval(): reset per-episode buffers
      - run_eval(env, obs, u, reward): record step-wise signals
      - post_eval(env, terminated, truncated): finalize per-episode metrics
      - evaluate_params(params): aggregate ONE JSON summary across episodes

    Env facts used:
      - obs = [x, theta, xdot, thetadot]
      - action u: continuous cart force (env clips to Box)
      - termination: |theta| > 0.2 rad; truncation: T == 1000
    """

    def __init__(self):
        self.EPISODES = 20
        self.HORIZON = 1000
        self.THETA_LIMIT = 0.2  # rad (health threshold)
        self.TOL_DEG = 5.0      # inner “stable” tube
        self._u_max = None  # filled on first run_eval


    # ---------------- helpers ----------------
    @staticmethod
    def _bias_index(arr: np.ndarray) -> float:
        denom = np.sum(np.abs(arr))
        return float(np.sum(arr) / denom) if denom > 1e-12 else 0.0

    @staticmethod
    def _rms(arr: np.ndarray) -> float:
        return float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0

    @staticmethod
    def _p95_abs(arr: np.ndarray) -> float:
        return float(np.percentile(np.abs(arr), 95)) if arr.size else 0.0

    @staticmethod
    def _zero_cross_rate(arr: np.ndarray) -> float:
        if arr.size <= 1:
            return 0.0
        s = np.sign(arr)
        return float(np.mean(s[1:] != s[:-1]))

    @staticmethod
    def _longest_streak(mask: np.ndarray) -> int:
        best = cur = 0
        for v in mask:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return int(best)

    @staticmethod
    def _default_failure_label(terminated: bool, truncated: bool) -> str:
        if terminated and not truncated:
            return "terminated"
        if truncated and not terminated:
            return "time-limit"
        if terminated and truncated:
            return "terminated+truncated"
        return "unknown"

    # ---------------- BaseStatistics API ----------------
    def initialise_eval(self):
        self.t = 0
        self.ep_return = 0.0
        self.x_list, self.th_list = [], []
        self.xd_list, self.omega_list = [], []
        self.u_list = []
        # u_max will be inferred on first run_eval

        self.per_ep = []

    def run_eval(self, env, obs, u, reward):
        # obs = [x, theta, xdot, thetadot]
        o = np.asarray(obs).ravel()
        x = float(o[0]) if o.size > 0 else 0.0
        th = float(o[1]) if o.size > 1 else 0.0
        xd = float(o[2]) if o.size > 2 else 0.0
        om = float(o[3]) if o.size > 3 else 0.0

        try:
            u_val = float(np.asarray(u).reshape(-1)[0])
        except Exception:
            u_val = float(u) if np.isscalar(u) else 0.0

        if self._u_max is None:
            try:
                self._u_max = float(np.asarray(env.action_space.high).ravel()[0])
            except Exception:
                self._u_max = 3.0

        self.x_list.append(x)
        self.th_list.append(th)
        self.xd_list.append(xd)
        self.omega_list.append(om)
        self.u_list.append(u_val)

        self.ep_return += float(reward)
        self.t += 1

    def post_eval(self, env, terminated, truncated):
        x = np.asarray(self.x_list, dtype=float)
        th = np.asarray(self.th_list, dtype=float)
        xd = np.asarray(self.xd_list, dtype=float)
        om = np.asarray(self.omega_list, dtype=float)
        u = np.asarray(self.u_list, dtype=float)

        length = int(self.t)
        ret = float(self.ep_return)
        failure_mode = self._default_failure_label(terminated, truncated)

        # Outcomes
        success = 1 if (truncated and not terminated) else 0  # reached horizon

        # Uprightness & bias
        upright_score = float(np.mean(np.cos(th))) if length else 0.0  # →1 when upright
        tilt_index = self._bias_index(th)                             # [-1,1]
        DEG = 180.0 / np.pi
        rms_theta_deg = self._rms(th * DEG)
        theta_p95_deg = self._p95_abs(th * DEG)

        healthy_mask = np.abs(th) < self.THETA_LIMIT
        healthy_frac = float(np.mean(healthy_mask)) if length else 0.0
        healthy_streak_max = self._longest_streak(healthy_mask.astype(bool))

        tol = self.TOL_DEG * np.pi / 180.0
        stable_mask = np.abs(th) < tol
        stable_frac = float(np.mean(stable_mask)) if length else 0.0
        stable_streak_max = self._longest_streak(stable_mask.astype(bool))

        # Cart centering
        drift_x_index = self._bias_index(x)  # [-1,1]
        rms_x = self._rms(x)
        rms_xdot = self._rms(xd)

        # Motion intensity
        rms_omega = self._rms(om)
        omega_p95 = self._p95_abs(om)
        zero_cross_rate_theta = self._zero_cross_rate(th)

        # Control / effort
        mean_abs_u = float(np.mean(np.abs(u))) if length else 0.0
        rms_u = self._rms(u)
        smoothness_u = float(np.mean(np.abs(np.diff(u)))) if length > 1 else 0.0
        sign_flip_rate_u = float(np.mean(np.sign(u[1:]) != np.sign(u[:-1]))) if length > 1 else 0.0
        sat_thresh = 0.9 * float(self._u_max if self._u_max is not None else 3.0)
        saturation_rate_u = float(np.mean(np.abs(u) > sat_thresh)) if length else 0.0

        self.per_ep.append({
            # outcomes
            "length": length,
            "return": ret,
            "failure_mode": failure_mode,
            "success": success,

            # uprightness & bias
            "upright_score": upright_score,
            "tilt_index": tilt_index,
            "rms_theta_deg": rms_theta_deg,
            "theta_p95_deg": theta_p95_deg,
            "healthy_frac": healthy_frac,
            "healthy_streak_max": healthy_streak_max,
            "stable_frac": stable_frac,
            "stable_streak_max": stable_streak_max,

            # cart centering
            "drift_x_index": drift_x_index,
            "rms_x": rms_x,
            "rms_xdot": rms_xdot,

            # motion intensity
            "rms_omega": rms_omega,
            "omega_p95": omega_p95,
            "zero_cross_rate_theta": zero_cross_rate_theta,

            # control / effort
            "mean_abs_u": mean_abs_u,
            "rms_u": rms_u,
            "smoothness_u": smoothness_u,
            "sign_flip_rate_u": sign_flip_rate_u,
            "saturation_rate_u": saturation_rate_u,
        })

    def evaluate_params(self, params: Sequence[float]) -> Dict[str, Any]:
        """
        Summarize ONE InvertedPendulum-v5 iteration (all episodes) into a compact JSON.

        Input
        -----
        params: Sequence[float]
            Kept for interface parity; not used here.

        Output JSON
        -----------
        {
          "meta": {
            "env": "InvertedPendulum-v5",
            "trials": <int>,         # episodes aggregated
            "horizon": 1000,         # step cap
            "theta_limit": 0.2       # env’s healthy bound (rad)
          },
          "failures": {
            "time_limit": <int>, "terminated": <int>, "terminated_truncated": <int>, "unknown": <int>,
            "time_limit_rate": <float>,        # fraction that reached horizon
            "success_rate": <float>            # same as time_limit_rate here
          },
          "stats": { "<metric>": {"median": <float>, "q1": <float>, "q3": <float>}, ... },
          "return_mean": <float>
        }

        Metrics (concise meanings)
          length, return                 : survival steps and reward sum (higher is better)
          success                        : 1 if reached horizon; else 0
          upright_score                  : mean cos(theta); →1 when upright
          tilt_index                     : signed lean in [-1,1] (− left, + right)
          rms_theta_deg, theta_p95_deg   : wobble magnitude and rare large tilts (deg)
          healthy_frac, healthy_streak_max : time and longest run with |theta| < 0.2 rad
          stable_frac, stable_streak_max : stricter tube |theta| < 5°
          drift_x_index                  : cart side bias in [-1,1]
          rms_x, rms_xdot                : cart displacement and jitter
          rms_omega, omega_p95           : angular velocity level and bursts
          zero_cross_rate_theta          : how often theta changes sign (oscillation)
          mean_abs_u, rms_u              : control magnitude / energy
          smoothness_u, sign_flip_rate_u : step-to-step action change / chatter
          saturation_rate_u              : fraction near actuation limits
        """
        df = pd.DataFrame(self.per_ep)
        if len(df) == 0:
            return {
                "meta": {"env": "InvertedPendulum-v5", "trials": 0, "horizon": self.HORIZON, "theta_limit": self.THETA_LIMIT},
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
            if arr.size:
                q1, med, q3 = np.percentile(arr, [25, 50, 75])
            else:
                q1 = med = q3 = np.nan
            stats[c] = {"median": float(med), "q1": float(q1), "q3": float(q3)}

        return {
            "meta": {"env": "InvertedPendulum-v5", "trials": len(df), "horizon": self.HORIZON, "theta_limit": self.THETA_LIMIT},
            "failures": failures,
            "stats": stats,
            "return_mean": float(df["return"].mean()),
        }
