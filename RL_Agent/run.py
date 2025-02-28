import os
import sys
import shutil
from datetime import datetime

# Paths
BASE_DIR = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent"
train_data = os.path.join(BASE_DIR, "data/train_data.json")
test_data = os.path.join(BASE_DIR, "data/test_data.json")
action_space = os.path.join(BASE_DIR, "config/action_space_config.json")
reward_config = os.path.join(BASE_DIR, "config/reward_config.json")

# Create experiment directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(BASE_DIR, f"experiments/experiment_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# Copy configurations
shutil.copy2(action_space, os.path.join(output_dir, 'action_space.json'))
shutil.copy2 (reward_config, os.path.join(output_dir, 'reward_config.json'))

# Import and prepare train script
from train_script import main as train_main
sys.argv = [
    'train_script.py',
    '--train_data', train_data,
    '--test_data', test_data,
    '--action_space', os.path.join(output_dir, 'action_space.json'),
    '--reward_config', os.path.join(output_dir, 'reward_config.json'),
    '--episodes', '1000',
    '--batch_size', '64',
    '--lr', '0.0003',
    '--output_dir', os.path.join(output_dir, 'training_output')
]

# Run training
train_main()

# Import and prepare analysis script
from ppo_analysis import main as analysis_main
sys.argv = [
    'ppo_analysis.py',
    '--metrics_file', os.path.join(output_dir, 'training_output', 'metrics.json'),
    '--output_dir', os.path.join(output_dir, 'analysis')
]

# Run analysis
analysis_main()
