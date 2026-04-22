import gymnasium as gym
import numpy as np
from gymnasium import spaces

class NetworkEnergyEnv(gym.Env):
    """
    A simulated 5G environment to train an RL agent to find optimal energy score weights.
    The agent dynamically adjusts the weights based on the current network state.
    """
    def __init__(self, dataset):
        super(NetworkEnergyEnv, self).__init__()
        # Dataset of shape (N, 5) -> latency, bandwidth, throughput, packet_loss, jitter
        self.dataset = dataset
        self.n_samples_max = len(self.dataset)
        self.current_step = 0
        
        # State: The metrics [latency, bandwidth, throughput, packet_loss, jitter]
        # Using a normalized range (-3 to 3 assuming standard scaled)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(5,), dtype=np.float32)
        
        # Action: Continuous weights [w_latency, w_bandwidth, w_throughput, w_packet_loss, w_jitter]
        # We allow weights between -1 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        state = self.dataset[self.current_step]
        info = {}
        return np.array(state, dtype=np.float32), info

    def step(self, action):
        state = self.dataset[self.current_step]
        
        # We define a reward based on a hypothetical objective:
        # We want the agent to learn to place high positive weights on 'good' metrics (bandwidth, throughput)
        # and negative weights on 'bad' metrics (latency, packet_loss, jitter) naturally without hardcoding it.
        # To simulate this, let's say the environment gives a positive signal when the 'Efficiency Score' 
        # (calculated by the weights) aligns with a generic QoS index.
        
        # QoS Index = + throughput + bandwidth - latency - packet_loss - jitter (assuming standardized data)
        # High QoS means good network condition.
        qos_index = state[1] + state[2] - state[0] - state[3] - state[4]
        
        # Agent's efficiency score calculated with its chosen actions (weights)
        agent_score = np.dot(action, state)
        
        # Reward: Agent is rewarded if its score tracks the QoS Index, but penalized for using unnecessarily large weights
        error = (qos_index - agent_score) ** 2
        weight_penalty = 0.1 * np.sum(np.abs(action))
        
        reward = -error - weight_penalty
        
        self.current_step += 1
        
        # Check if episode is done
        terminated = bool(self.current_step >= self.n_samples_max)
        truncated = False
        
        info = {}
        if terminated:
            # Wrap around randomly for continuous learning
            next_state = self.dataset[0]
        else:
            next_state = self.dataset[self.current_step]
            
        return np.array(next_state, dtype=np.float32), reward, terminated, truncated, info
