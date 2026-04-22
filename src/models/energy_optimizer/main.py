import os
from data_utils import load_data, preprocess_data
from model import train_model, evaluate_model
from predict import simulate_real_time_optimization
from sklearn.model_selection import train_test_split
# Suppress warnings for cleaner lab output
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    print("="*50)
    print("5G Core Network Energy Optimizer - RL Initialization")
    print("="*50)
    
    # 1. Load and preprocess data
    data_path = os.path.join(os.path.dirname(__file__), "../../data/data.csv")
    print(f"Loading network data from {data_path}...")
    data = load_data(data_path)
    
    # We no longer need the static add_energy_efficiency_label() 
    # as the RL agent dynamically learns weights based on QoS reward!
    
    # Mock 'y' array so we don't have to overhaul preprocess_data 
    if 'energy_label' not in data.columns:
        data['energy_label'] = 'MockLabel' 
        
    X, y, scaler, encoder = preprocess_data(data)

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data Split: {len(X_train)} train samples, {len(X_test)} test samples.\n")

    # 3. Train the RL model
    rl_model, env = train_model(X_train, y_train)
    
    # 4. Evaluate the RL model
    evaluate_model(rl_model, env, X_test, y_test, encoder)

    # 5. Simulate real-time optimization for presentation
    simulate_real_time_optimization(rl_model, scaler, encoder)
