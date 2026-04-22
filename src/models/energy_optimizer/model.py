import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from rl_env import NetworkEnergyEnv

def train_model(X_train, y_train=None):
    """
    Train an RL agent using PPO on the custom NetworkEnergyEnv.
    y_train is kept in the signature to match previous pipeline, but ignored.
    """
    print("Setting up RL Environment...")
    env = NetworkEnergyEnv(X_train)
    
    # Check if the environment is valid
    check_env(env, warn=True)
    
    print("Training RL Agent (PPO)...")
    # Using PPO for training our continuous action space
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001)
    
    # Train for 5000 timesteps as a quick simulation for the lab
    model.learn(total_timesteps=5000)
    print("RL Agent Training Complete.")
    
    return model, env

def evaluate_model(model, env_dummy, X_test, y_test=None, encoder=None):
    """
    Evaluate the RL agent by running episodes on the test dataset.
    """
    print("\n--- RL Agent Evaluation on Test Set ---")
    test_env = NetworkEnergyEnv(X_test)
    obs, info = test_env.reset()
    
    total_rewards = []
    
    for _ in range(min(100, len(X_test))):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_rewards.append(reward)
        if terminated:
            break
            
    print(f"Average RL Reward on Test Data: {np.mean(total_rewards):.4f}")
    return np.mean(total_rewards)
