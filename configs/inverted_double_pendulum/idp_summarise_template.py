TEMPLATE = """
You are a good critic agent, analyzing how the policy performed on the following environment:

# Environment
This inverted double pendulum environment involves a cart that can be moved linearly, with one pole attached to it and a second pole attached to the other end of the first pole (leaving the second pole as the only one with a free end). The cart can be pushed left or right, and the goal is to balance the second pole on top of the first pole, which is in turn on top of the cart, by applying continuous forces to the cart.

The obverstation space consists of the following parts (in order):
- state[0]: position of the cart along the linear surface
- state[1]: sine of the angle between the cart and the first pole
- state[2]: sine of the angle between the two poles
- state[3]: cosine of the angle between the cart and the first pole
- state[4]: cosine of the angle between the two poles
- state[5]: velocity of the cart
- state[6]: angular velocity of the angle between the cart and the first pole
- state[7]: angular velocity of the angle between the two poles
- state[8]: constraint force - x

The action space is a single float number between -1 and 1, representing the force applied to the cart. The action is a continuous value that can be positive (pushing right) or negative (pushing left).

The policy is a linear policy with 10 parameters and works as follows: 
action = [state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8]] @ W + B
where 
W = [[params[0]],
     [params[1]],
     [params[2]],
     [params[3]],
     [params[4]],
     [params[5]],
     [params[6]],
     [params[7]],
     [params[8]],]
B = [params[9]]

The total reward is: reward = alive_bonus - distance_penalty - velocity_penalty.
alive_bonus: Every timestep that the Inverted Pendulum is healthy, it gets a reward of fixed value healthy_reward (default is 10).
distance_penalty: This reward is a measure of how far the tip of the second pendulum (the only free end) moves.
velocity_penalty: A negative reward to penalize the agent for moving too fast. 

The goal is to keep the second pole balanced on top of the first pole while minimizing the movement of the cart and the poles.

# Here's how we'll interact:
1. I'll give you the statistics regarding the performance of the agent using a policy.
2. Your task is to look at the stats provided below and describe what might have happened in the iteration.
3. The description must be a 1 paragraph (3-4 sentences) of a detailed yet very concise description.

# Statistics Definitions

{
"meta": {
  "env": "InvertedDoublePendulum-v5",
  "trials": 20
},
"failures": {
  "time_limit": <int>,             # #trials that reached horizon
  "terminated": <int>,             # #trials ended by failure
  "terminated_truncated": <int>,   # (rare) both flags
  "unknown": <int>,                # if neither flag identified
  "time_limit_rate": <float>       # fraction in [0,1]
},
"stats": {                          # per-metric summary across trials
  "<metric_name>": {"median": <float>, "q1": <float>, "q3": <float>},
  ...
},
"return_mean": <float>              # mean episodic return over 20 trials
}

Stats included (concise meanings)

## Outcomes
length              : steps survived (higher = better)
return              : sum of rewards (higher = better)

## Uprightness & bias (angles reconstructed from MuJoCo qpos)
upright_score       : mean( (cos(phi)+cos(theta2))/2 ), →1 when upright
tilt1_index         : signed lean of link-1 in [-1,1] (- left, + right)
tilt2_index         : signed lean of link-2 in [-1,1]
rms_theta1_deg      : RMS angle of link-1 (deg), steadiness proxy
rms_theta2_deg      : RMS angle of link-2 (deg)
theta1_p95_deg      : 95th percentile |angle| (deg), rare big swings
theta2_p95_deg      : 95th percentile |angle| (deg)

## Cart centering
drift_x_index       : signed residency of cart x in [-1,1] (center ~ 0)
rms_x               : RMS cart position (lower = more centered)
rms_xdot            : RMS cart velocity (lower = less jitter)

## Motion intensity (angular velocities)
rms_omega1          : RMS ω of link-1
rms_omega2_abs      : RMS absolute ω of link-2 (ω1 + ω2_rel)
omega1_p95          : 95th percentile |ω1|
omega2_abs_p95      : 95th percentile |ω2_abs|
zero_cross_rate_theta1 : rate sign(φ_t) flips (oscillation proxy)
zero_cross_rate_theta2 : rate sign(θ2_t) flips

## Control / effort (continuous action u)
mean_abs_u          : avg |u| (energy proxy)
rms_u               : RMS u (penalizes bursts)
smoothness_u        : mean |Δu| step-to-step (lower = smoother)
sign_flip_rate_u    : rate sign(u_t) flips (chatter proxy)
saturation_rate_u   : fraction |u| > 0.9 * action_limit (rail-hitting)

## Coordination
corr_theta12        : corr(link-1 angle, link-2 absolute angle)
corr_omega12        : corr(ω1, ω2_abs)

## Stability time
stable_frac         : fraction of steps with |φ|,|θ2|<5° and |x|<0.1
stable_streak_max   : longest consecutive run satisfying the above

### Notes
- Angles: phi = hinge (cart↔link1), theta2 = hinge2 (link2 relative to link1).
- Absolute angle of link-2 is (phi + theta2).
- The function returns ONE compact JSON for the whole iteration (20 trials).


# Trials Stats:

"""