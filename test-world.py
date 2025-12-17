import gymnasium as gym
import numpy as np
import re
import argparse

def parse_parameters(input_text):
    s = input_text.split("\n")[0]
    pattern = re.compile(r"params\[(\d+)\]\s*[:=]\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)")
    matches = pattern.findall(s)

    if not matches:
        pattern = re.compile(r"[+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?")
        matches = pattern.findall(s)
        results = [float(m) for m in matches]
    else:
        results = [float(m[1]) for m in matches]
    print(results)
    return np.array(results).reshape(-1)

def compute(params):
    env = gym.make("InvertedPendulum-v5")
    total_reward = []
    params = parse_parameters(params)
    W = params[0:4].reshape(4, 1)  # Weights for the 4 state dimensions
    b = params[4]                  # Bias term

    for _ in range(20):
        state, info = env.reset()
        rewards = 0.0
        done = False
        while not done:
            action = np.argmax(np.dot(state, W) + b)
            action = np.array([action])
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards += reward
        print(rewards)

        total_reward.append(rewards)

    print("Total Reward:", np.mean(total_reward))
    env.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=str,
        help="Path to the config file",
    )
    args = parser.parse_args()
    compute(args.params)
