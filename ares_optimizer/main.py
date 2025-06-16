"""
Main orchestrator for Ares Optimizer.
Drives the entire RL training loop and manages the optimization process.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
import torch

from ares_optimizer.trainer import AresTrainer
from ares_optimizer.ares_env.code_executor import CodeExecutor
from ares_optimizer.ares_env.test_manager import TestManager
from ares_optimizer.state_representation.feature_extractor import FeatureExtractor
from ares_optimizer.reward_calculator import RewardCalculator
from ares_optimizer.rl_agent.agent import DQNAgent
from ares_optimizer.utils.logger import setup_logger

class AresOptimizer:
    def __init__(
        self,
        functions_dir: str = "functions_to_optimize",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        num_episodes: int = 1000,
        max_steps_per_episode: int = 10,
        save_freq: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Ares Optimizer.

        Args:
            functions_dir: Directory containing functions to optimize
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory to save logs
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum transformations per episode
            save_freq: Frequency of model checkpointing
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.functions_dir = Path(functions_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_freq = save_freq
        self.device = device

        # Create necessary directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # Setup logging
        self.logger = setup_logger(
            name="ares_optimizer",
            log_dir=self.log_dir,
            level=logging.INFO
        )

        # Initialize components
        self.code_executor = CodeExecutor()
        self.test_manager = TestManager()
        self.trainer = AresTrainer(
            code_executor=self.code_executor,
            test_manager=self.test_manager,
            device=self.device,
            save_dir=str(self.checkpoint_dir)
        )

        self.logger.info("Ares Optimizer initialized successfully")

    def load_functions(self) -> List[Tuple[str, Dict]]:
        """
        Load functions and their test cases from the functions directory.

        Returns:
            List of (code, test_cases) tuples
        """
        functions = []
        for file_path in self.functions_dir.glob("*.py"):
            try:
                with open(file_path, "r") as f:
                    code = f.read()
                
                # Load corresponding test file
                test_file = file_path.with_suffix(".json")
                if test_file.exists():
                    with open(test_file, "r") as f:
                        test_cases = json.load(f)
                else:
                    self.logger.warning(f"No test cases found for {file_path}")
                    continue

                functions.append((code, test_cases))
                self.logger.info(f"Loaded function from {file_path}")

            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                continue

        return functions

    def train(self, functions: List[Tuple[str, Dict]], verbose: bool = True):
        """
        Train the optimizer on the provided functions.

        Args:
            functions: List of (code, test_cases) tuples
            verbose: Whether to print progress
        """
        self.logger.info(f"Starting training on {len(functions)} functions")
        self.trainer.train(
            code_samples=functions,
            num_episodes=self.num_episodes,
            save_freq=self.save_freq,
            verbose=verbose
        )
        self.logger.info("Training completed")

    def optimize_function(
        self,
        code: str,
        test_cases: Dict,
        verbose: bool = True
    ) -> Tuple[str, Dict]:
        """
        Optimize a single function using the trained model.

        Args:
            code: Function code to optimize
            test_cases: Test cases for the function
            verbose: Whether to print progress

        Returns:
            Tuple of (optimized_code, optimization_report)
        """
        self.logger.info("Starting function optimization")
        
        # Get initial performance
        function_name = self._extract_function_name(code)
        orig_runtime, orig_memory, _ = self.code_executor.execute_function(
            code, function_name, test_cases["test_cases"]
        )

        # Optimize the code
        optimized_code = self.trainer.optimize_code(
            code,
            test_cases,
            max_steps=self.max_steps_per_episode,
            verbose=verbose
        )

        # Get optimized performance
        opt_runtime, opt_memory, _ = self.code_executor.execute_function(
            optimized_code,
            function_name,
            test_cases["test_cases"]
        )

        # Calculate improvements (handle zero division)
        runtime_improvement = 0.0
        memory_improvement = 0.0
        
        if orig_runtime > 0:
            runtime_improvement = (orig_runtime - opt_runtime) / orig_runtime * 100
        if orig_memory > 0:
            memory_improvement = (orig_memory - opt_memory) / orig_memory * 100

        # Generate optimization report
        report = {
            "original": {
                "runtime_ms": orig_runtime,
                "memory_mb": orig_memory,
                "code": code
            },
            "optimized": {
                "runtime_ms": opt_runtime,
                "memory_mb": opt_memory,
                "code": optimized_code
            },
            "improvements": {
                "runtime_reduction": runtime_improvement,
                "memory_reduction": memory_improvement
            }
        }

        self.logger.info("Function optimization completed")
        return optimized_code, report

    def _extract_function_name(self, code: str) -> str:
        """Extract the first function name from the code string."""
        import re
        match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        if match:
            return match.group(1)
        raise ValueError("No function definition found in code.")

def main():
    """Main entry point for the Ares Optimizer."""
    import argparse
    parser = argparse.ArgumentParser(description="Ares Optimizer")
    parser.add_argument(
        "--functions_dir",
        default="functions_to_optimize",
        help="Directory containing functions to optimize"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--log_dir",
        default="logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Maximum transformations per episode"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100,
        help="Frequency of model checkpointing"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress during training"
    )

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = AresOptimizer(
        functions_dir=args.functions_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        save_freq=args.save_freq,
        device=args.device
    )

    # Load functions
    functions = optimizer.load_functions()
    if not functions:
        print("No functions found to optimize")
        return

    # Train the optimizer
    optimizer.train(functions, verbose=args.verbose)

    # Optimize each function
    for code, test_cases in functions:
        optimized_code, report = optimizer.optimize_function(
            code,
            test_cases,
            verbose=args.verbose
        )
        
        # Print optimization report
        print("\nOptimization Report:")
        print(f"Runtime Improvement: {report['improvements']['runtime_reduction']:.2f}%")
        print(f"Memory Improvement: {report['improvements']['memory_reduction']:.2f}%")
        print("\nOriginal Code:")
        print(report['original']['code'])
        print("\nOptimized Code:")
        print(report['optimized']['code'])

if __name__ == "__main__":
    main() 