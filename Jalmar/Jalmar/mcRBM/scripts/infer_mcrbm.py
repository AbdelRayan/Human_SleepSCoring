#!/usr/bin/env python3
"""
Inference script for mcRBM model - extract and analyze latent states.
Usage: python infer_mcrbm.py --model-dir ./experiments/mcrbm_exp1/weights --exp-dir ./experiments/mcrbm_exp1
"""

import argparse
import os
import sys
from infer_states import mcRBMInference


def main():
    parser = argparse.ArgumentParser(description='Infer latent states from trained mcRBM')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing trained model weights')
    parser.add_argument('--exp-dir', type=str, required=True,
                       help='Experiment directory containing data')
    parser.add_argument('--model-file', type=str, default='ws_final.mat',
                       help='Trained model filename')
    parser.add_argument('--states-file', type=str, default='states.mat',
                       help='States file (if available)')
    parser.add_argument('--states-dir', type=str, default='./',
                       help='Directory containing states file')
    parser.add_argument('--analyze', action='store_true',
                       help='Run full analysis after inference')
    
    args = parser.parse_args()
    
    try:
        print(f"Model directory: {args.model_dir}")
        print(f"Experiment directory: {args.exp_dir}")
        
        # Initialize inference engine
        inference = mcRBMInference(
            modelDir=args.model_dir,
            expDir=args.exp_dir,
            modelFile=args.model_file
        )
        
        # Load data
        print("Loading data...")
        inference.loadData(args.states_dir, args.states_file)
        
        # Compute states
        print("Computing latent states...")
        states = inference.computeStates(saveProbabilities=True, saveBinary=True)
        
        print(f"Computed {states['num_unique_states']} unique states" if 'num_unique_states' in states else "States computed")
        
        # Optional analysis
        if args.analyze:
            print("Running analysis...")
            analysis = inference.analyzeStates(saveResults=True)
            
            print("Computing transition matrix...")
            trans_matrix = inference.computeTransitionMatrix()
            
            print("Visualizing state sequence...")
            state_seq = inference.visualizeStateSequence(maxSamples=5000, saveResults=True)
            
            print("Analysis complete!")
        
        print("Inference complete! Check analysis/ directory for results.")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
