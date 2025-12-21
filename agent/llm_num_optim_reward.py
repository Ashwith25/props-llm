from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from agent.policy.linear_policy import LinearPolicy
from agent.policy.replay_buffer import EpisodeRewardBuffer
from agent.policy.replay_buffer import ReplayBuffer
from agent.policy.llm_brain_reward import LLMBrainReward
from world.base_world import BaseWorld
from jinja2 import Template
import numpy as np
import re
import time
import json
import random
from stats.base_statistics import BaseStatistics

class LLMNumOptimRewardAgent:
    def __init__(
        self,
        logdir,
        dim_action,
        dim_state,
        max_traj_count,
        max_traj_length,
        max_best_length,
        memory_strategy,
        summary: bool,
        summary_template: Template,
        summary_desc_file,
        stats: BaseStatistics,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        bias,
        optimum,
        search_step_size,
        env_desc_file=None,
    ):
        self.start_time = time.process_time()
        self.api_call_time = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.bias = bias
        self.optimum = optimum
        self.search_step_size = search_step_size
        self.env_desc_file = env_desc_file
        self.summary = summary
        self.summary_template = summary_template
        self.summary_desc_file = summary_desc_file
        self.stats=stats
        with open("ip-params-sampled.txt", "r") as f:
            self.ip_params = [np.array(l.split(" | ")[0].split(",")).astype(float) for l in f.readlines()]
        
        random.shuffle(self.ip_params)

        if not self.bias:
            param_count = dim_action * dim_state
        else:
            param_count = dim_action * dim_state + dim_action
        self.rank = param_count

        self.policy = LinearPolicy(dim_actions=dim_action, dim_states=dim_state)
        self.replay_buffer = EpisodeRewardBuffer(max_size=max_traj_count)
        self.traj_buffer = ReplayBuffer(max_traj_count, max_traj_length)
        self.llm_brain = LLMBrainReward(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0

        if self.bias:
            self.dim_state += 1

    def rollout_episode(self, world: BaseWorld, logging_file, record=True):
        state = world.reset()
        state = np.expand_dims(state, axis=0)
        logging_file.write(
            f"{', '.join([str(x) for x in self.policy.get_parameters().reshape(-1)])}\n"
        )
        logging_file.write(f"parameter ends\n\n")
        logging_file.write(f"state | action | reward\n")
        terminated, truncated = False, False
        step_idx = 0
        if record:
            self.traj_buffer.start_new_trajectory()
        if self.summary:
            self.stats.initialise_eval()
        while not (terminated or truncated):
            action = self.policy.get_action(state.T)
            action = np.reshape(action, (1, self.dim_action))
            if world.discretize:
                action = np.argmax(action)
                action = np.array([action])
            next_state, reward, terminated, truncated = world.step(action)
            logging_file.write(f"{state.T[0]} | {action[0]} | {reward}\n")
            if record:
                self.traj_buffer.add_step(state, action, reward)
            if self.summary:
                self.stats.run_eval(world.env, state, action, reward)
            state = next_state
            step_idx += 1
            self.total_steps += 1
        if self.summary:
            self.stats.post_eval(world.env, terminated, truncated)
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        self.total_episodes += 1
        return world.get_accu_reward()

    def random_warmup(self, world: BaseWorld, logdir, num_episodes):
        for episode in range(num_episodes):
            # self.policy.initialize_policy()
            # ! epsilon-decay kinda initialization
            if episode < 10:
                self.policy.initialize_policy()
            else:
                self.policy.update_policy(self.ip_params[self.training_episodes % len(self.ip_params)])
            # Run the episode and collect the trajectory
            print(f"Rolling out warmup episode {episode}...")
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w")
            # result = self.rollout_episode(world, logging_file)

            results = []
            for idx in range(self.num_evaluation_episodes):
                if idx == 0:
                    result = self.rollout_episode(world, logging_file, record=True)
                else:
                    result = self.rollout_episode(world, logging_file, record=False)
                results.append(result)
            print(f"Results: {results}")
            result = np.mean(results)
            if self.summary:
                RESP = self.stats.evaluate_params(self.policy.get_parameters())
                prompt = self.summary_template.render(
                    {
                        "env_description": self.env_desc_file,
                        "stats_definitions": self.summary_desc_file,
                        "trials_stats": json.dumps(RESP) 
                    }
                )
                # print("Prompt for summary:", prompt)
                explanation = self.llm_brain.query_reasoning_llm(prompt)
                self.replay_buffer.add(
                    np.array(self.policy.get_parameters()).reshape(-1), world.get_accu_reward(), explanation
                )
                logging_file.write(f"\nExplanation: {explanation}\n")
            else:
                self.replay_buffer.add(
                    np.array(self.policy.get_parameters()).reshape(-1), world.get_accu_reward()
                )

            logging_file.close()
            print(f"Result: {result}")
        # print(self.replay_buffer.buffer)
        # self.replay_buffer.sort()

    def train_policy(self, world: BaseWorld, logdir):

        def str_nd_examples(replay_buffer, traj_buffer: ReplayBuffer, n):
            text = ""
            print('Num trajs in buffer:', len(traj_buffer.buffer))
            print('Num params in buffer:', len(replay_buffer.buffer))
            for parameters, true_reward, pred_reward, _ in replay_buffer.buffer:
                l = ""
                for i in range(n):
                    l += f"params[{i}]: {parameters[i]:.5g}; "
                l += f"true_reward(params): {true_reward:.2f}\n"
                l += f"predicted_reward(params): {pred_reward:.2f}\n" if pred_reward else f"predicted_reward(params): N/A\n"
                #! uncomment for summary
                # if explanation:
                #     l += f"Episodic performance details: {explanation}\n\n"
                # l += f"Trajectory: {traj_buffer.buffer[idx]}\n\n"
                text += l
            return text

        # Update the policy using llm_brain, q_table and replay_buffer
        print("Updating the policy...")
        print("Current Policy Parameters:")
        print(self.dim_state, self.dim_action)

        # rand_weight = np.round((np.random.rand(self.dim_state, self.dim_action) - 0.5) * 12, 1)
        # rand_bias = np.round((np.random.rand(1, self.dim_action) - 0.5) * 12, 1)
        # params = np.concatenate((rand_weight, rand_bias), axis=0)

        # self.policy.initialize_policy()
        # params = np.array(self.policy.get_parameters()).reshape(-1)

        params = self.ip_params[self.training_episodes % len(self.ip_params)]
        print("Params before LLM call:", params, "Params shape:", params.shape)

        _, prev_true, prev_predicted, prev_confidence = self.replay_buffer.buffer[-1]

        pred_reward, confidence, reason, reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim_semantics(
            prev_true,
            prev_predicted,
            prev_confidence,
            params,
            str_nd_examples(self.replay_buffer, self.traj_buffer, self.rank),
            self.training_episodes,
            self.env_desc_file,
            self.rank,
            self.optimum,
            self.search_step_size
        )
        self.api_call_time += api_time

        print(self.policy.get_parameters().shape)
        # print(new_parameter_list.shape)
        self.policy.update_policy(params)
        print(self.policy.get_parameters().shape)
        logging_q_filename = f"{logdir}/parameters.txt"
        logging_q_file = open(logging_q_filename, "w")
        logging_q_file.write(str(self.policy))
        logging_q_file.close()
        q_reasoning_filename = f"{logdir}/parameters_reasoning.txt"
        q_reasoning_file = open(q_reasoning_filename, "w")
        q_reasoning_file.write(reasoning)
        q_reasoning_file.close()
        print("Policy updated!")

        # Run the episode and collect the trajectory
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        results = []
        for idx in range(self.num_evaluation_episodes):
            if idx == 0:
                result = self.rollout_episode(world, logging_file, record=True)
            else:
                result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        print(f"Results: {results}")
        result = np.mean(results)

        
        if self.summary:
            RESP = self.stats.evaluate_params(params)
            prompt = self.summary_template.render(
                {
                    "env_description": self.env_desc_file,
                    "stats_definitions": self.summary_desc_file,
                    "trials_stats": json.dumps(RESP) 
                }
            )
            # print("Prompt for summary:", prompt)
            explanation=self.llm_brain.query_reasoning_llm(prompt)
            # ! Uncomment the below code while implementing summary
            # self.replay_buffer.add(new_parameter_list, result, explanation)
        else:
            self.replay_buffer.add(params, result, pred_reward, confidence)

        self.training_episodes += 1

        _cpu_time = time.process_time() - self.start_time
        _api_time = self.api_call_time
        _total_episodes = self.total_episodes
        _total_steps = self.total_steps
        _total_reward = result
        return _cpu_time, _api_time, _total_episodes, _total_steps, _total_reward, pred_reward, confidence
    

    def evaluate_policy(self, world: BaseWorld, logdir):
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        return results
