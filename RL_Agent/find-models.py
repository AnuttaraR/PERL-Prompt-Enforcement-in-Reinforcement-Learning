import os
import argparse
import json


def find_ppo_model_paths(base_dir):
    """Find PPO model paths in the given directory structure"""
    model_paths = {}

    # Find the main model
    experiment_dir = os.path.join(base_dir, "experiment_20250226_220826")
    if os.path.exists(experiment_dir):
        # Look for training output dirs
        training_dirs = [d for d in os.listdir(experiment_dir)
                         if os.path.isdir(os.path.join(experiment_dir, d)) and d.startswith("training_output")]

        for training_dir in training_dirs:
            final_model_dir = os.path.join(experiment_dir, training_dir, "final_model")
            if os.path.exists(final_model_dir):
                model_paths["main_ppo_model"] = final_model_dir
                break

    # Find ablation models
    ablation_prefix = "action_ablation_study_"
    ablation_dirs = [d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(ablation_prefix)]

    for ablation_dir in ablation_dirs:
        ablation_path = os.path.join(base_dir, ablation_dir)

        # Check for ablation types
        for ablation_type in ["action_what_only", "action_how_only", "action_if_can_only", "action_minimal",
                              "action_unified", "baseline"]:
            # Try direct path
            direct_path = os.path.join(ablation_path, ablation_type)
            if os.path.isdir(direct_path):
                # Check for final_model
                final_model_dir = os.path.join(direct_path, "final_model")
                if os.path.exists(final_model_dir):
                    model_paths[f"ppo_{ablation_type}"] = final_model_dir
                    continue

            # Try timestamped directories
            timestamped_dirs = [d for d in os.listdir(ablation_path)
                                if os.path.isdir(os.path.join(ablation_path, d)) and d.startswith(ablation_type + "_")]

            for ts_dir in timestamped_dirs:
                final_model_dir = os.path.join(ablation_path, ts_dir, "final_model")
                if os.path.exists(final_model_dir):
                    model_paths[f"ppo_{ablation_type}"] = final_model_dir
                    break

    return model_paths


def find_dpo_model_paths(base_dir):
    """Find DPO model paths in the given directory structure"""
    model_paths = {}

    # Find the main model
    dpo_run_dir = os.path.join(base_dir, "dpo_run_20250307_155639")
    if os.path.exists(dpo_run_dir):
        dpo_model_dir = os.path.join(dpo_run_dir, "dpo_model")
        if os.path.exists(dpo_model_dir):
            model_paths["main_dpo_model"] = dpo_model_dir

    # Find ablation models
    ablation_prefix = "dpo_ablation_study_"
    ablation_dirs = [d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(ablation_prefix)]

    for ablation_dir in ablation_dirs:
        ablation_path = os.path.join(base_dir, ablation_dir)

        # Check for ablation types
        for ablation_type in ["action_diversity_reward", "balanced_pairs", "higher_temperature", "beta_low",
                              "beta_high"]:
            # Try direct path
            direct_path = os.path.join(ablation_path, ablation_type)
            if os.path.isdir(direct_path):
                # Check for final_model
                final_model_dir = os.path.join(direct_path, "final_model")
                if os.path.exists(final_model_dir):
                    model_paths[f"dpo_{ablation_type}"] = final_model_dir
                    continue

            # Try timestamped directories
            timestamped_dirs = [d for d in os.listdir(ablation_path)
                                if os.path.isdir(os.path.join(ablation_path, d)) and d.startswith(ablation_type + "_")]

            for ts_dir in timestamped_dirs:
                final_model_dir = os.path.join(ablation_path, ts_dir, "final_model")
                if os.path.exists(final_model_dir):
                    model_paths[f"dpo_{ablation_type}"] = final_model_dir
                    break

    return model_paths


def main():
    parser = argparse.ArgumentParser(description='Find model paths in your directory structure')
    parser.add_argument('--ppo_base_dir', type=str,
                        default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/ppo_results/experiments',
                        help='Base directory for PPO models')
    parser.add_argument('--dpo_base_dir', type=str, default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent',
                        help='Base directory for DPO models')
    parser.add_argument('--output_file', type=str, default='model_paths.json',
                        help='Output file to save model paths')

    args = parser.parse_args()

    # Find model paths
    ppo_paths = find_ppo_model_paths(args.ppo_base_dir)
    dpo_paths = find_dpo_model_paths(args.dpo_base_dir)

    # Combine results
    all_paths = {
        "ppo_models": ppo_paths,
        "dpo_models": dpo_paths
    }

    # Print findings
    print("\nFound PPO models:")
    for name, path in ppo_paths.items():
        print(f"  {name}: {path}")

    print("\nFound DPO models:")
    for name, path in dpo_paths.items():
        print(f"  {name}: {path}")

    # Save to file
    with open(args.output_file, 'w') as f:
        json.dump(all_paths, f, indent=2)

    print(f"\nModel paths saved to {args.output_file}")
    print(f"Total models found: {len(ppo_paths) + len(dpo_paths)}")


if __name__ == "__main__":
    main()