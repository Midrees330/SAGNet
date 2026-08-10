import os
import torch
import argparse
from collections import OrderedDict
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import time

from models.SAGNet import SAGNet
from utils.data_loader import make_datapath_list, ImageDataset, ImageTransform

torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_parser():
    parser = argparse.ArgumentParser(description='SAGNet Testing')
    parser.add_argument('-l', '--load', type=str, default='630')
    parser.add_argument('-o', '--out_path', type=str, default='./test_results')
    parser.add_argument('--image_size', type=int, default=286)
    parser.add_argument('--crop_size', type=int, default=256)
    return parser


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict


def unnormalize(x):
    """Unnormalize from [-1, 1] to [0, 1]"""
    return (x * 0.5 + 0.5).clamp(0, 1)

def create_output_dirs(out_path):
    """Create output directory structure"""
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, 'shadow_free'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'comparison_grids'), exist_ok=True)


def load_model_checkpoint(checkpoint_path, model, device):
    """Load model checkpoint with debug information"""
    try:
        if os.path.isfile(checkpoint_path):
            print(f"\n{'='*60}")
            print("CHECKPOINT VERIFICATION")
            print(f"{'='*60}")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, dict):
                print("Checkpoint type: Dictionary")
                print(f"Keys: {list(checkpoint.keys())}")
                if 'epoch' in checkpoint:
                    print(f"Trained epochs: {checkpoint['epoch']}")
                if 'best_psnr' in checkpoint:
                    print(f"Best PSNR: {checkpoint['best_psnr']:.2f} dB")
                if 'optimizer_state_dict' in checkpoint:
                    print("Contains optimizer state: Yes")
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                    
                first_key = list(state_dict.keys())[0]
                first_value = state_dict[first_key]
                print(f"\nFirst weight '{first_key}':")
                print(f"  Shape: {first_value.shape}")
                print(f"  Mean: {first_value.mean():.6f}")
                print(f"  Std: {first_value.std():.6f}")
                print(f"  Min/Max: [{first_value.min():.6f}, {first_value.max():.6f}]")
            else:
                print("Checkpoint type: State dict only")
                state_dict = checkpoint
                first_key = list(state_dict.keys())[0]
                first_value = state_dict[first_key]
                print(f"First weight '{first_key}': mean={first_value.mean():.6f}, std={first_value.std():.6f}")
            
            print(f"{'='*60}\n")
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(fix_model_state_dict(checkpoint['model_state_dict']))
            else:
                model.load_state_dict(fix_model_state_dict(checkpoint))
            
            print(f"{'='*60}")
            print("MODEL ARCHITECTURE CHECK")
            print(f"{'='*60}")
            
            first_param = next(model.parameters())
            print("First model parameter:")
            print(f"  Shape: {first_param.shape}")
            print(f"  Mean: {first_param.mean():.6f}")
            print(f"  Std: {first_param.std():.6f}")
            print(f"  Min/Max: [{first_param.min():.6f}, {first_param.max():.6f}]")
            
            if first_param.abs().max() < 0.0001:
                print("WARNING: Model weights are near zero!")
            else:
                print("Model weights loaded successfully")
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("\nModel parameters:")
            print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
            print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
            
            print(f"{'='*60}\n")
            
            return True
        else:
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            return False
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def test(model, test_dataset, device, args):
    """Test the model and save results"""
    model.eval()
    
    create_output_dirs(args.out_path)
    
    print(f"\n{'='*80}")
    print("TESTING")
    print(f"{'='*80}")
    print(f"Output directory: {args.out_path}")
    print(f"Testing {len(test_dataset)} images...")
    print(f"{'='*80}\n")
    
    total_time = 0.0
    
    progress_bar = tqdm(range(len(test_dataset)), desc="Processing")
    
    with torch.no_grad():
        for n in progress_bar:
            shadow_img, _, gt = test_dataset[n]
            
            filename = os.path.basename(test_dataset.img_list['path_A'][n])
            base_name = os.path.splitext(filename)[0]
            
            shadow_img_batch = shadow_img.unsqueeze(0).to(device)
            gt_batch = gt.unsqueeze(0)
            
            start_time = time.time()
            outputs = model(shadow_img_batch)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            shadow_free = outputs['output']
            
            shadow_free_cpu = shadow_free.cpu()
            shadow_img_cpu = shadow_img_batch.cpu()
            gt_cpu = gt_batch.cpu()
            
            progress_bar.set_postfix({
                'Time': f'{inference_time*1000:.1f}ms',
                'Avg': f'{(total_time/(n+1))*1000:.1f}ms'
            })
            
            shadow_free_pil = transforms.ToPILImage()(unnormalize(shadow_free_cpu[0]))
            shadow_free_pil.save(os.path.join(args.out_path, 'shadow_free', filename))
            
            comparison = torch.cat([
                unnormalize(shadow_img_cpu), 
                unnormalize(gt_cpu), 
                unnormalize(shadow_free_cpu)
            ], dim=0)
            save_image(
                comparison,
                os.path.join(args.out_path, 'comparison_grids', f'{base_name}_comparison.jpg'),
                nrow=3,
                padding=10,
                pad_value=1.0
            )
    
    avg_time = total_time / len(test_dataset)
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"Total images: {len(test_dataset)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {avg_time*1000:.2f}ms per image")
    print(f"FPS: {1.0/avg_time:.2f}")
    print(f"{'='*80}\n")
    
    print(f"Results saved to: {args.out_path}/")
    print("  - shadow_free/         : Shadow-free output images")
    print("  - comparison_grids/    : Comparisons [Shadow | GT | Output]")
    print(f"{'='*80}\n")


def main(parser):
    print("\n" + "="*80)
    print("SAIFormer Testing")
    print("="*80 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = SAGNet(
        input_channels=3,
        output_channels=3,
        embed_dim=64,
        num_blocks=[2, 2, 2, 2],
        num_heads=[2, 4, 8, 16]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    if parser.load.isdigit():
        checkpoint_name = f'SAGNet_{parser.load}'
    elif not parser.load.startswith('SAGNet_'):
        if parser.load == 'best':
            checkpoint_name = 'SAGNet_best'
        else:
            checkpoint_name = f'SAGNet_{parser.load}'
    else:
        checkpoint_name = parser.load
    
    checkpoint_path = f'./checkpoints/{checkpoint_name}.pth'
    print(f"Loading checkpoint: {checkpoint_name}")
    print(f"Path: {checkpoint_path}")
    
    if not load_model_checkpoint(checkpoint_path, model, device):
        return
    
    model = model.to(device)
    
    mean = (0.5,)
    std = (0.5,)
    
    test_img_list = make_datapath_list(phase='test')
    test_dataset = ImageDataset(
        img_list=test_img_list,
        img_transform=ImageTransform(size=parser.image_size, crop_size=parser.crop_size, mean=mean, std=std),
        phase='test_no_crop'
    )
    
    print(f"Test images: {len(test_dataset)}")
    
    test(model, test_dataset, device, parser)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)