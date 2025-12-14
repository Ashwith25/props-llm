from typing import Sequence, Dict, Any

class BaseStatistics:
    def evaluate_params(self, params: Sequence[float]) -> Dict[str, Any]:
        raise NotImplementedError("run method not implemented")
    
    def initialise_eval(self):
        raise NotImplementedError("interpolate_state method not implemented")

    def run_eval(self, env, obs, u, reward):
        raise NotImplementedError("discretize_state method not implemented")

    def post_eval(self, env, terminated, truncated):
        raise NotImplementedError("discretize_state method not implemented")

