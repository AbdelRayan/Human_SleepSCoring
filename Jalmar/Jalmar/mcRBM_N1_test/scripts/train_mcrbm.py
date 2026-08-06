#!/usr/bin/env python3
"""
Training entry point for mcRBM.

Usage:
    python train_mcrbm.py --config-dir ./configuration_files --exp-config exp_config.ini --model-config model_config.ini
"""

import argparse
import os
import sys
from mcrbm import mcRBM


def main():
    parser = argparse.ArgumentParser(description='Train mcRBM model')
    parser.add_argument('--config-dir', type=str, default='./configuration_files',
                       help='Directory containing configuration files')
    parser.add_argument('--exp-config', type=str, default='exp_config.ini',
                       help='Experiment configuration filename')
    parser.add_argument('--model-config', type=str, default='model_config.ini',
                       help='Model configuration filename')
    
    args = parser.parse_args()
    
    # Initialize and train model
    print(f"Configuration directory: {args.config_dir}")
    print(f"Experiment config: {args.exp_config}")
    print(f"Model config: {args.model_config}")
    
    try:
        model = mcRBM(
            refDir=args.config_dir,
            expConfigFilename=args.exp_config,
            modelConfigFilename=args.model_config
        )
        
        print("Starting mcRBM training...")
        model.train()
        print("Training complete!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
