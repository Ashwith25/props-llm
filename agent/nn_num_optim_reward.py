from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from agent.policy.linear_policy import LinearPolicy
from agent.policy.replay_buffer import EpisodeRewardBuffer
from agent.policy.nn_brain_reward import NNBrainReward # Import the new Brain
from world.base_world import BaseWorld
import numpy as np
import time
import random
import os

class NNNumOptimRewardAgent:
    def __init__(
        self,
        logdir,
        dim_action,
        dim_state,
        max_traj_count,
        # Keep unused args to match the runner's config signature
        llm_si_template=None,
        llm_model_name=None,
        warmup_episodes=20,
        num_evaluation_episodes=10,
        bias=True,
        optimum=0,
        dataset_file=None,
        env_desc_file=None,
    ):
        self.start_time = time.process_time()
        self.total_episodes = 0
        self.total_steps = 0 
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.bias = bias
        self.num_evaluation_episodes = num_evaluation_episodes
        self.logdir = logdir
        self.iteration = 0

        # Initialize Policy
        if not self.bias:
            param_count = dim_action * dim_state
            self.policy = LinearPolicyNoBias(dim_actions=dim_action, dim_states=dim_state)
        else:
            param_count = dim_action * dim_state + dim_action
            self.policy = LinearPolicy(dim_actions=dim_action, dim_states=dim_state)
        
        self.rank = param_count
        self.replay_buffer = EpisodeRewardBuffer(max_size=max_traj_count)
        
        # Load Sampled Params (Same as LLM agent)
        with open(dataset_file, 'r') as f:
            self.sampled_params = [np.array(i.split(' | ')[0].split(",")).astype(float) for i in f.readlines()]
        random.shuffle(self.sampled_params)
        self.warmup_samples = random.sample(self.sampled_params, warmup_episodes)
        
        # --- INITIALIZE NEURAL NETWORK ---
        self.nn_brain = NNBrainReward(input_dim=self.rank, epochs=50, lr=0.001, device="cuda")

    def rollout_episode(self, world: BaseWorld, logging_file=None):
        # ... (Same as your existing rollout_episode code) ...
        state = world.reset()
        if logging_file:
            logging_file.write(f"state | action | reward\n")

        done = False
        truncated = False
        while not (done or truncated):
            state_in = np.expand_dims(state, axis=0)
            action = self.policy.get_action(state_in.T)
            action = np.reshape(action, (1, self.dim_action))
            
            if world.discretize:
                action_env = np.argmax(action)
                action_env = np.array([action_env])
            else:
                action_env = action
            next_state, reward, done, truncated = world.step(action_env)
            logging_file.write(f"{state.T[0]} | {action[0]} | {reward}\n")

            state = next_state
            self.total_steps += 1
        
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        self.total_episodes += 1
        return world.get_accu_reward()

    def evaluate_current_policy(self, world, logdir=None):
        # ... (Same as your existing evaluate_current_policy code) ...
        results = []
        log_file = None
        if logdir:
            logging_filename = f"{logdir}/training_rollout.txt"
            log_file = open(logging_filename, "w")

        print(f"Rolling out episode {self.iteration}...")
        for _ in range(self.num_evaluation_episodes):
            if log_file:
                params = self.policy.get_parameters().reshape(-1)
                params_str = ", ".join([str(x) for x in params])
                log_file.write(f"{params_str}\nparameter ends\n\n")

            result = self.rollout_episode(world, logging_file=log_file)
            
            if log_file:
                log_file.write("\n")
            results.append(result)
            
        if log_file:
            log_file.close()
        return np.mean(results)

    def random_warmup(self, world, logdir, warmup_episodes):
        # ... (Same as your existing code, ensures buffer is populated) ...
        print(f"--- Starting Warmup for {warmup_episodes} trials ---")
        for i in range(warmup_episodes):
            self.policy.update_policy(self.warmup_samples[i])
            params = self.policy.get_parameters().reshape(-1)
            
            logging_filename = f"{logdir}/warmup_rollout_{i}.txt"
            rewards = []
            
            with open(logging_filename, "w") as f:
                for _ in range(self.num_evaluation_episodes):
                    params_str = ", ".join([str(x) for x in params])
                    f.write(f"{params_str}\nparameter ends\n\n")
                    reward = self.rollout_episode(world, logging_file=f)
                    rewards.append(reward)
                    f.write("\n")
            
            avg_reward = np.mean(rewards)
            # Add to buffer (None for pred/confidence since it's warmup)
            self.replay_buffer.add(params, avg_reward, None, None)
            print(f"Warmup {i+1}/{warmup_episodes}: Avg Reward {avg_reward:.2f}")
        print("--- Warmup Complete ---")

    def train_policy(self, world: BaseWorld, episode_logdir):
        # 1. Select Parameters (Same sampling logic as LLM agent)
        target_params = self.sampled_params[self.iteration % len(self.sampled_params)]

        # --- DIFFERENCE STARTS HERE ---
        
        # 2. Train the Neural Network on current History
        # We re-train or fine-tune every step to simulate ICL "seeing" more data
        self.nn_brain.train(self.replay_buffer)

        # 3. Predict Reward using NN
        pred_reward, confidence, reasoning = self.nn_brain.predict(target_params)
        
        # Mock API Time (NN is instant compared to LLM)
        api_time = 0.05 

        # --- LOGGING ---
        # Write dummy reasoning so your parser doesn't break
         # 4. Execute Environment
        self.policy.update_policy(target_params)

        with open(f"{episode_logdir}/reward_reasoning.txt", "w") as f:
            f.write(f"system:\nNN Training on {len(self.replay_buffer.buffer)} samples.\n\nLLM:\n{reasoning}")

        with open(f"{episode_logdir}/parameters.txt", "w") as f:
            f.write(str(self.policy))
       
        true_reward = self.evaluate_current_policy(world, logdir=episode_logdir)

        # 5. Update Buffer
        self.replay_buffer.add(target_params, true_reward, pred_reward, confidence)

        self.iteration += 1

        _cpu_time = time.process_time() - self.start_time
        _api_time = api_time
        _total_episodes = self.total_episodes
        _total_steps = self.total_steps
        _total_reward = true_reward
        _pred_reward = pred_reward
        _confidence = confidence

        return target_params,_cpu_time, _api_time, _total_episodes, _total_steps, _total_reward, _pred_reward, _confidence