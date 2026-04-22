import numpy as np
import time

def simulate_real_time_optimization(model, scaler, encoder=None, num_samples=5):
    """
    Simulate real-time energy efficiency optimization in the lab.
    Shows the RL Agent dynamically picking weights for different network conditions.
    """
    print("\n" + "="*50)
    print(">>> Real-Time RL Energy Efficiency Optimizer <<<")
    print("="*50 + "\n")
    
    # We will run real-time random conditions through the RL Agent
    for i in range(num_samples):
        # Generate a random network condition: [latency, bandwidth, throughput, packet_loss, jitter]
        sample = np.random.uniform(low=[10, 10, 1, 0, 0], high=[100, 1000, 500, 5, 30], size=(1, 5))
        sample_scaled = scaler.transform(sample)
        
        # RL Agent predicts the optimal weights for this condition based on its training
        obs = sample_scaled[0]
        action, _states = model.predict(obs, deterministic=True)
        
        print(f"Step {i+1}: New Network Condition Detected")
        print(f"   [Lat: {sample[0][0]:.1f}ms | BW: {sample[0][1]:.1f}Mbps | Thrpt: {sample[0][2]:.1f}Mbps | Loss: {sample[0][3]:.1f}% | Jit: {sample[0][4]:.1f}ms]")
        
        print(f"RL Agent taking Action (Generating Weights)...")
        time.sleep(1) # Small delay for lab presentation effect
        
        # Display the weights dynamically allocated by the RL agent
        weights = action
        print(f"   => W_Latency:    {weights[0]:+.3f}")
        print(f"   => W_Bandwidth:  {weights[1]:+.3f}")
        print(f"   => W_Throughput: {weights[2]:+.3f}")
        print(f"   => W_Loss:       {weights[3]:+.3f}")
        print(f"   => W_Jitter:     {weights[4]:+.3f}")
        
        # Calculate dynamic energy score
        dynamic_score = np.dot(obs, weights)
        print(f"Energy/QoS Index Realized: {dynamic_score:+.3f}")
        print("-" * 50)
        time.sleep(1)

