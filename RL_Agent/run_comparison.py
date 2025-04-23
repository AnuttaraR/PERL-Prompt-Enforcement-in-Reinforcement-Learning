import os
import sys
import subprocess
import json
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("run_comparison")


def run_command(cmd, description):
    """Run a command and log output without streaming it"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        # Run the process with output capturing instead of streaming
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Use text mode with default encoding
            errors='replace'  # Replace problematic characters
        )

        # Log only summary instead of full output
        logger.info(f"Process completed with return code: {result.returncode}")

        if result.returncode != 0:
            logger.error(f"Command failed with return code {result.returncode}")
            # Log just the error summary, not the full output
            logger.error(f"Error summary: {result.stderr.splitlines()[-5:] if result.stderr else 'No error output'}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error running command: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run complete comparative analysis workflow')
    parser.add_argument('--test_data', type=str,
                        default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/test_data.json',
                        help='Path to test dataset')
    parser.add_argument('--action_space', type=str,
                        default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/action_space_config.json',
                        help='Path to action space configuration')
    parser.add_argument('--reward_config', type=str,
                        default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/reward_config.json',
                        help='Path to reward configuration')
    parser.add_argument('--ppo_dir', type=str,
                        default='run_results/ppo/main/model/final_ppo_model',
                        help='Path to main PPO model')
    parser.add_argument('--dpo_dir', type=str,
                        default='run_results/dpo/main/model/dpo_model',
                        help='Path to main DPO model')
    parser.add_argument('--ppo_ablation_dir', type=str,
                        default='run_results/ppo/ablations',
                        help='Directory containing PPO ablation results')
    parser.add_argument('--dpo_ablation_dir', type=str,
                        default='run_results/dpo/ablations',
                        help='Directory containing DPO ablation results')
    parser.add_argument('--llm_models', type=str, nargs='+', default=['gpt-3.5-turbo', 'gpt-4'],
                        help='LLM models to use for evaluation')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Size of test subset to use (for quicker testing)')
    parser.add_argument('--baseline_only', action='store_true',
                        help='Run only baseline RAG evaluation')
    parser.add_argument('--ablation_only', action='store_true',
                        help='Run only ablation analysis')
    parser.add_argument('--complete_only', action='store_true',
                        help='Run only complete comparison')

    args = parser.parse_args()

    # Create timestamped output directory for all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"comparative_analysis_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)

    # Define output directories for each step
    baseline_dir = os.path.join(base_output_dir, "baseline_results")
    ablation_dir = os.path.join(base_output_dir, "ablation_analysis")
    comparison_dir = os.path.join(base_output_dir, "complete_comparison")

    # Create output directories
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(ablation_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    python_executable = sys.executable

    # 1. Run baseline RAG evaluation
    # if not args.ablation_only and not args.complete_only:
    #     logger.info("=" * 60)
    #     logger.info("STEP 1: Running baseline RAG evaluation")
    #     logger.info("=" * 60)
    #
    #     baseline_cmd = [
    #         python_executable,
    #         "baseline_rag.py",
    #         "--test_data", args.test_data,
    #         "--output_dir", baseline_dir
    #     ]
    #
    #     # Add LLM models
    #     baseline_cmd.extend(["--llm_models"])
    #     baseline_cmd.extend(args.llm_models)
    #
    #     # Add subset size if specified
    #     if args.subset_size:
    #         baseline_cmd.extend(["--subset_size", str(args.subset_size)])
    #
    #     success = run_command(baseline_cmd, "Baseline RAG evaluation")
    #
    #     if not success:
    #         logger.error("Baseline RAG evaluation failed")
    #         if not args.baseline_only:
    #             logger.info("Continuing with other steps...")

    # 2. Run ablation analysis
    if not args.baseline_only and not args.complete_only:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Running ablation analysis")
        logger.info("=" * 60)

        ablation_cmd = [
            python_executable,
            "ablation_analysis.py",
            "--output_dir", ablation_dir,
            "--ppo_experiment_dir", args.ppo_ablation_dir,
            "--dpo_experiment_dir", args.dpo_ablation_dir
        ]

        success = run_command(ablation_cmd, "Ablation analysis")

        if not success:
            logger.error("Ablation analysis failed")
            if not args.ablation_only:
                logger.info("Continuing with other steps...")

    # 3. Run complete comparative analysis
    if not args.baseline_only and not args.ablation_only:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Running complete comparative analysis")
        logger.info("=" * 60)

        comparison_cmd = [
            python_executable,
            "comparative_analysis.py",
            "--test_data", args.test_data,
            "--action_space", args.action_space,
            "--reward_config", args.reward_config,
            "--ppo_dir", args.ppo_dir,
            "--dpo_dir", args.dpo_dir,
            "--ppo_ablation_dir", args.ppo_ablation_dir,
            "--dpo_ablation_dir", args.dpo_ablation_dir,
            "--output_dir", comparison_dir,
            "--eval_all"
        ]

        # Add LLM models
        comparison_cmd.extend(["--llm_models"])
        comparison_cmd.extend(args.llm_models)

        # Add subset size if specified
        if args.subset_size:
            comparison_cmd.extend(["--subset_size", str(args.subset_size)])

        success = run_command(comparison_cmd, "Complete comparative analysis")

        if not success:
            logger.error("Complete comparative analysis failed")

    # Create a summary file with links to all results
    summary = {
        "timestamp": timestamp,
        "baseline_results_dir": baseline_dir,
        "ablation_analysis_dir": ablation_dir,
        "complete_comparison_dir": comparison_dir,
        "test_data": args.test_data,
        "llm_models": args.llm_models,
        "subset_size": args.subset_size
    }

    with open(os.path.join(base_output_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"Comparative analysis workflow completed")
    logger.info(f"All results saved to: {base_output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()