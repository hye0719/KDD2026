"""
Main Entry Point for Metalens FNO Project

Usage:
    # Train models
    python main.py train --model fno --dim 1d
    python main.py train --model unet --dim 2d
    python main.py train --model specboost --dim 2d
    
    # Evaluate models
    python main.py eval --checkpoint checkpoints/xxx/fno_1d_best.pth --model fno --dim 1d
    
    # Visualize results
    python main.py viz --checkpoint checkpoints/xxx/fno_1d_best.pth --model fno --dim 1d
"""

import argparse
import yaml
import sys
import os

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='Metalens FNO Project')
    parser.add_argument('command', type=str, choices=['train', 'eval', 'viz'],
                        help='Command to run: train, eval, or viz')
    parser.add_argument('--config', type=str, default='config/configure.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        choices=['fno', 'linear', 'unet', 'specboost'],
                        help='Model type')
    parser.add_argument('--dim', type=str, default=None,
                        choices=['1d', '2d'],
                        help='Dimension (1d or 2d)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (for eval and viz)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for visualization')
    parser.add_argument('--k', type=int, default=None,
                        help='Number of top samples to save (for viz)')
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 오버라이드 적용
    model_type = args.model if args.model else config['model']['type']
    dim = args.dim if args.dim else config['model']['dim']
    
    print("=" * 60)
    print(f"🚀 Metalens FNO Project")
    print(f"   Command: {args.command}")
    print(f"   Model: {model_type.upper()}")
    print(f"   Dimension: {dim.upper()}")
    print("=" * 60)
    
    if args.command == 'train':
        if model_type == 'specboost':
            from train_specboost import train_specboost
            train_specboost(config, override_dim=dim)
        else:
            from train import train
            train(config, override_model=model_type, override_dim=dim)
    
    elif args.command == 'eval':
        if args.checkpoint is None:
            print("❌ Error: --checkpoint is required for eval command")
            return
        from evaluation import evaluate
        evaluate(config, args.checkpoint, model_type=model_type, dim=dim)
    
    elif args.command == 'viz':
        if args.checkpoint is None:
            print("❌ Error: --checkpoint is required for viz command")
            return
        
        from utils import get_device
        from data.dataset import get_dataloaders
        from visualization import (
            load_model,
            save_top_k_results,
            save_top_k_results_specboost,
            get_specboost_models
        )
        import torch
        
        device = get_device(config['general'].get('device', 'auto'))
        dataset, _, val_loader = get_dataloaders(config, dim=dim)

        output_dir = args.output if args.output else './viz_results'
        os.makedirs(output_dir, exist_ok=True)
        
        k = args.k if args.k else config['output'].get('top_k_samples', 10)
        
        if model_type == "specboost":
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model_0, model_1 = get_specboost_models(config, dim=dim)
            model_0.load_state_dict(checkpoint['model_0'])
            model_1.load_state_dict(checkpoint['model_1'])
            model_0 = model_0.to(device)
            model_1 = model_1.to(device)
            
            save_top_k_results_specboost(
                config=config,
                model_0=model_0,
                model_1=model_1,
                val_loader=val_loader,
                dataset=dataset,
                device=device,
                dim=dim,
                result_dir=output_dir,
                k=k
            )
        else:
            model = load_model(config, args.checkpoint, model_type, dim, device)
            save_top_k_results(
                config=config,
                model=model,
                val_loader=val_loader,
                dataset=dataset,
                device=device,
                dim=dim,
                model_type=model_type,
                result_dir=output_dir,
                k=k
            )


if __name__ == "__main__":
    main()
