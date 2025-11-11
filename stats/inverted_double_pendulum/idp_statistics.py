from typing import Sequence, Dict, Any
from stats.base_statistics import BaseStatistics
import numpy as np
import pandas as pd
import gymnasium as gym
import cv2
from pathlib import Path

class IDP_Statistics(BaseStatistics):
    def __init__(self):
        self.EPISODES = 20
        self.TOL_DEG = 5.0
        self.TOL_X = 0.1
        self.DEG = 180.0 / np.pi

    # ---------------- helpers ----------------
    def _bias_index(self, arr: np.ndarray) -> float:
        denom = np.sum(np.abs(arr))
        return float(np.sum(arr) / denom) if denom > 1e-12 else 0.0

    def _rms(self, arr: np.ndarray) -> float:
        return float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0

    def _p95_abs(self, arr: np.ndarray) -> float:
        return float(np.percentile(np.abs(arr), 95)) if arr.size else 0.0

    def _zero_cross_rate(self, arr: np.ndarray) -> float:
        if arr.size <= 1: return 0.0
        s = np.sign(arr)
        return float(np.mean(s[1:] != s[:-1]))

    def _longest_streak(self, mask: np.ndarray) -> int:
        best = cur = 0
        for v in mask:
            if v: cur += 1; best = max(best, cur)
            else: cur = 0
        return int(best)

    def _default_failure_label(self, terminated: bool, truncated: bool) -> str:
        if terminated and not truncated: return "terminated"
        if truncated  and not terminated: return "time-limit"
        if terminated and truncated:      return "terminated+truncated"
        return "unknown"

    def _extract_kinematics(self, env: gym.Env, obs) -> Dict[str, float]:
        # Prefer MuJoCo qpos/qvel: DoF [x, hinge(phi), hinge2(theta2)]
        try:
            un = env.unwrapped
            while hasattr(un, "env"):
                un = un.env
            data = getattr(un, "data", None) or getattr(un, "sim", None).data
            qpos = np.asarray(data.qpos).ravel()
            qvel = np.asarray(data.qvel).ravel()
            x       = float(qpos[0])
            phi     = float(qpos[1])
            theta2  = float(qpos[2])
            xdot    = float(qvel[0])
            omega1  = float(qvel[1])
            omega2r = float(qvel[2])
            return dict(
                x=x, xdot=xdot, phi=phi, omega1=omega1,
                theta2=theta2, omega2_rel=omega2r,
                phi_abs=phi, theta2_abs=phi+theta2
            )
        except Exception:
            o = np.asarray(obs).ravel()
            return dict(x=float(o[0]) if o.size>0 else 0.0, xdot=0.0,
                        phi=0.0, omega1=0.0, theta2=0.0, omega2_rel=0.0,
                        phi_abs=0.0, theta2_abs=0.0)

    def _med_iqr(self, arr: np.ndarray):
        if arr.size == 0: return float("nan"), (float("nan"), float("nan"))
        q1, q2, q3 = np.percentile(arr, [25, 50, 75])
        return float(q2), (float(q1), float(q3))

    def _summarize_numeric(self, df: pd.DataFrame, cols) -> Dict[str, Dict[str, float]]:
        out = {}
        for c in cols:
            med, (q1, q3) = self._med_iqr(df[c].to_numpy(dtype=float))
            out[c] = {"median": med, "q1": q1, "q3": q3}
        return out

    def initialise_eval(self):
        self.ep_reward = 0.0
        self.u_list = []
        self.x_list, self.xdot_list = [], []
        self.phi_list, self.theta2_list = [], []
        self.omega1_list, self.omega2r_list, self.omega2a_list = [], [], []
        self.t = 0
        self.per_ep = []

    def run_eval(self, env, obs, u, reward):
        kin = self._extract_kinematics(env, obs)
        self.x_list.append(kin["x"])
        self.xdot_list.append(kin["xdot"])
        self.phi_list.append(kin["phi"])
        self.theta2_list.append(kin["theta2"])
        self.omega1_list.append(kin["omega1"])
        self.omega2r_list.append(kin["omega2_rel"])
        self.omega2a_list.append(kin["omega1"] + kin["omega2_rel"])
        self.u_list.append(float(u))

        self.ep_reward += float(reward)
        self.t += 1

    def post_eval(self, env, terminated, truncated):
        length = self.t
        ret = self.ep_reward
        failure_mode = self._default_failure_label(terminated, truncated)

        phi_arr = np.asarray(self.phi_list)
        th2_arr = np.asarray(self.theta2_list)
        x_arr = np.asarray(self.x_list)
        xdot_arr = np.asarray(self.xdot_list)
        w1_arr = np.asarray(self.omega1_list)
        w2a_arr = np.asarray(self.omega2a_list)
        u_arr = np.asarray(self.u_list, dtype=float)

        upright_score = float(np.mean((np.cos(phi_arr) + np.cos(th2_arr)) / 2.0)) if length>0 else 0.0
        tilt1_index   = self._bias_index(phi_arr)
        tilt2_index   = self._bias_index(th2_arr)
        rms_theta1_deg = self._rms(phi_arr * self.DEG)
        rms_theta2_deg = self._rms(th2_arr * self.DEG)
        theta1_p95_deg = self._p95_abs(phi_arr * self.DEG)
        theta2_p95_deg = self._p95_abs(th2_arr * self.DEG)

        drift_x_index = self._bias_index(x_arr)
        rms_x   = self._rms(x_arr)
        rms_xdot = self._rms(xdot_arr)

        rms_omega1      = self._rms(w1_arr)
        rms_omega2_abs  = self._rms(w2a_arr)
        omega1_p95      = self._p95_abs(w1_arr)
        omega2_abs_p95  = self._p95_abs(w2a_arr)
        zcr_theta1      = self._zero_cross_rate(phi_arr)
        zcr_theta2      = self._zero_cross_rate(th2_arr)

        mean_abs_u   = float(np.mean(np.abs(u_arr))) if length>0 else 0.0
        rms_u        = self._rms(u_arr)
        smoothness_u = float(np.mean(np.abs(np.diff(u_arr)))) if length>1 else 0.0
        sign_flip_rate_u = float(np.mean(np.sign(u_arr[1:]) != np.sign(u_arr[:-1]))) if length>1 else 0.0
        try:
            u_max = float(np.asarray(env.action_space.high).ravel()[0])
            sat_thresh = 0.9 * u_max
            saturation_rate_u = float(np.mean(np.abs(u_arr) > sat_thresh)) if length>0 else 0.0
        except Exception:
            saturation_rate_u = 0.0

        try:
            corr_theta12 = float(np.corrcoef(phi_arr, phi_arr + th2_arr)[0,1]) if length>1 else 0.0
        except Exception:
            corr_theta12 = 0.0
        try:
            corr_omega12 = float(np.corrcoef(w1_arr, w2a_arr)[0,1]) if length>1 else 0.0
        except Exception:
            corr_omega12 = 0.0

        mask_stable = (np.abs(phi_arr*self.DEG) < self.TOL_DEG) & \
                    (np.abs(th2_arr*self.DEG) < self.TOL_DEG) & \
                    (np.abs(x_arr) < self.TOL_X)
        stable_frac = float(np.mean(mask_stable)) if length>0 else 0.0
        stable_streak_max = self._longest_streak(mask_stable.astype(bool))

        self.per_ep.append({
            "length": length, "return": ret, "failure_mode": failure_mode,
            "upright_score": upright_score,
            "tilt1_index": tilt1_index, "tilt2_index": tilt2_index,
            "rms_theta1_deg": rms_theta1_deg, "rms_theta2_deg": rms_theta2_deg,
            "theta1_p95_deg": theta1_p95_deg, "theta2_p95_deg": theta2_p95_deg,
            "drift_x_index": drift_x_index, "rms_x": rms_x, "rms_xdot": rms_xdot,
            "rms_omega1": rms_omega1, "rms_omega2_abs": rms_omega2_abs,
            "omega1_p95": omega1_p95, "omega2_abs_p95": omega2_abs_p95,
            "zero_cross_rate_theta1": zcr_theta1, "zero_cross_rate_theta2": zcr_theta2,
            "mean_abs_u": mean_abs_u, "rms_u": rms_u, "smoothness_u": smoothness_u,
            "sign_flip_rate_u": sign_flip_rate_u, "saturation_rate_u": saturation_rate_u,
            "corr_theta12": corr_theta12, "corr_omega12": corr_omega12,
            "stable_frac": stable_frac, "stable_streak_max": stable_streak_max
        })

    def evaluate_params(self, params: Sequence[float]) -> Dict[str, Any]:
        """
        Tool: Evaluate a linear policy on InvertedDoublePendulum-v5 and return a
        single-iteration JSON summary.

        Policy
        u = state @ W + b, where W is built from the first len(params)-1 values
        (auto-truncated or zero-padded to obs_dim) and b is the last value.

        Input
        params (length=10, floats): Linear policy coefficients.
            - params[:9] -> observation weights (W)
            - params[9]  -> scalar bias (b)

        Output (JSON)
        {
            "meta": {
            "env": "InvertedDoublePendulum-v5",
            "episodes": 20,
            "tol_deg": 5.0,      # angle tolerance for stability checks
            "tol_x": 0.1         # cart position tolerance for stability checks
            },
            "failures": {
            "time_limit": <int>,             # #episodes that reached horizon
            "terminated": <int>,             # #episodes ended by failure
            "terminated_truncated": <int>,   # (rare) both flags
            "unknown": <int>,                # if neither flag identified
            "time_limit_rate": <float>       # fraction in [0,1]
            },
            "stats": {                          # per-metric summary across episodes
            "<metric_name>": {"median": <float>, "q1": <float>, "q3": <float>},
            ...
            },
            "return_mean": <float>              # mean episodic return over 20 episodes
        }

        Stats included (concise meanings)
        # Outcomes
        length              : steps survived (higher = better)
        return              : sum of rewards (higher = better)

        # Uprightness & bias (angles reconstructed from MuJoCo qpos)
        upright_score       : mean( (cos(phi)+cos(theta2))/2 ), →1 when upright
        tilt1_index         : signed lean of link-1 in [-1,1] (- left, + right)
        tilt2_index         : signed lean of link-2 in [-1,1]
        rms_theta1_deg      : RMS angle of link-1 (deg), steadiness proxy
        rms_theta2_deg      : RMS angle of link-2 (deg)
        theta1_p95_deg      : 95th percentile |angle| (deg), rare big swings
        theta2_p95_deg      : 95th percentile |angle| (deg)

        # Cart centering
        drift_x_index       : signed residency of cart x in [-1,1] (center ~ 0)
        rms_x               : RMS cart position (lower = more centered)
        rms_xdot            : RMS cart velocity (lower = less jitter)

        # Motion intensity (angular velocities)
        rms_omega1          : RMS ω of link-1
        rms_omega2_abs      : RMS absolute ω of link-2 (ω1 + ω2_rel)
        omega1_p95          : 95th percentile |ω1|
        omega2_abs_p95      : 95th percentile |ω2_abs|
        zero_cross_rate_theta1 : rate sign(φ_t) flips (oscillation proxy)
        zero_cross_rate_theta2 : rate sign(θ2_t) flips

        # Control / effort (continuous action u)
        mean_abs_u          : avg |u| (energy proxy)
        rms_u               : RMS u (penalizes bursts)
        smoothness_u        : mean |Δu| step-to-step (lower = smoother)
        sign_flip_rate_u    : rate sign(u_t) flips (chatter proxy)
        saturation_rate_u   : fraction |u| > 0.9 * action_limit (rail-hitting)

        # Coordination
        corr_theta12        : corr(link-1 angle, link-2 absolute angle)
        corr_omega12        : corr(ω1, ω2_abs)

        # Stability time
        stable_frac         : fraction of steps with |φ|,|θ2|<5° and |x|<0.1
        stable_streak_max   : longest consecutive run satisfying the above

        Notes
        - Angles: phi = hinge (cart↔link1), theta2 = hinge2 (link2 relative to link1).
        - Absolute angle of link-2 is (phi + theta2).
        - The function returns ONE compact JSON for the whole iteration (20 episodes).
        """
        df = pd.DataFrame(self.per_ep)
        fail_counts = df["failure_mode"].value_counts()
        failures = {
            "time_limit": int(fail_counts.get("time-limit", 0)),
            "terminated": int(fail_counts.get("terminated", 0)),
            "terminated_truncated": int(fail_counts.get("terminated+truncated", 0)),
            "unknown": int(fail_counts.get("unknown", 0)),
            "time_limit_rate": float(fail_counts.get("time-limit", 0) / len(df)) if len(df) else 0.0,
        }

        numeric_cols = [c for c in df.columns if c != "failure_mode"]
        # summarize each numeric column with median, Q1, Q3
        stats = {}
        for c in numeric_cols:
            arr = df[c].to_numpy(dtype=float)
            q1, med, q3 = np.percentile(arr, [25, 50, 75]) if arr.size else (np.nan, np.nan, np.nan)
            stats[c] = {"median": float(med), "q1": float(q1), "q3": float(q3)}

        result = {
            "meta": {"env": "InvertedDoublePendulum-v5", "trials": self.EPISODES},
            "failures": failures,
            "stats": stats,
            "return_mean": float(df["return"].mean()) if len(df) else float("nan"),
        }
        return result
