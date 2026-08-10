from utils.data_loader import make_datapath_list, ImageDataset, ImageTransform
from models.SAGNet import SAGNet
from torchvision.utils import make_grid, save_image
from collections import OrderedDict
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time
import torch
import os
import math

torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        window = self.window
        channel = img1.size(1)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class MultiScalePerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(2)])
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(2, 7)])
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(7, 12)])
        self.slice4 = nn.Sequential(*[vgg[x] for x in range(12, 21)])
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        # Proper normalization to [0, 1] then to ImageNet stats
        pred = (pred * 0.5) + 0.5
        target = (target * 0.5) + 0.5
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # Clamp to prevent extreme values in FP16
        pred = torch.clamp(pred, -10, 10)
        target = torch.clamp(target, -10, 10)
        
        with torch.no_grad():
            target_f1 = self.slice1(target)
            target_f2 = self.slice2(target_f1)
            target_f3 = self.slice3(target_f2)
            target_f4 = self.slice4(target_f3)
        
        pred_f1 = self.slice1(pred)
        pred_f2 = self.slice2(pred_f1)
        pred_f3 = self.slice3(pred_f2)
        pred_f4 = self.slice4(pred_f3)
        
        # Safe loss computation with clamping
        loss = (torch.clamp(F.l1_loss(pred_f1, target_f1), 0, 100) + 
                torch.clamp(F.l1_loss(pred_f2, target_f2), 0, 100) + 
                torch.clamp(F.l1_loss(pred_f3, target_f3), 0, 100) + 
                torch.clamp(F.l1_loss(pred_f4, target_f4), 0, 100)) / 4.0
        
        return loss


class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0,  0,  0],
             [1,  2,  1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def get_edges(self, x):
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)
        edge_x = F.conv2d(x, sobel_x, padding=1, groups=3)
        edge_y = F.conv2d(x, sobel_y, padding=1, groups=3)
        # Increased epsilon for numerical stability in FP16
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-4)  # was 1e-6
        return edges

    def forward(self, pred, target):
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        # Safe loss with clamping
        loss = torch.clamp(F.l1_loss(pred_edges, target_edges), 0, 100)
        return loss


class ColorConstancyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, input_img):
        pred_diff = pred - input_img
        target_diff = target - input_img
        
        pred_mean = torch.mean(pred, dim=[2, 3], keepdim=True)
        target_mean = torch.mean(target, dim=[2, 3], keepdim=True)
        
        diff_loss = F.l1_loss(pred_diff, target_diff)
        mean_loss = F.l1_loss(pred_mean, target_mean)
        
        return diff_loss + mean_loss


class FrequencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Input clamping for stability
        pred = torch.clamp(pred, -1, 1)
        target = torch.clamp(target, -1, 1)
        
        # FFT with error handling - Force FP32 for stability
        try:
            with torch.cuda.amp.autocast(enabled=False):
                pred_fp32 = pred.float()
                target_fp32 = target.float()
                
                pred_fft = torch.fft.fft2(pred_fp32)
                target_fft = torch.fft.fft2(target_fp32)
                
                # Magnitude spectrum
                pred_mag = torch.abs(pred_fft)
                target_mag = torch.abs(target_fft)
                
                # Clamp magnitudes to prevent extreme values
                pred_mag = torch.clamp(pred_mag, 0, 1000)
                target_mag = torch.clamp(target_mag, 0, 1000)
                
                loss = F.l1_loss(pred_mag, target_mag)
                loss = torch.clamp(loss, 0, 100)  # Safety clamp
                
                return loss
        except Exception as e:
            # Fallback to zero loss if FFT fails
            print(f"WARNING: FrequencyLoss FFT failed: {e}, returning zero loss")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


class EnhancedLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = MultiScalePerceptualLoss(device=device).to(device)
        self.edge_loss = EdgeAwareLoss()
        self.color_loss = ColorConstancyLoss()
        self.freq_loss = FrequencyLoss()

    def forward(self, outputs, target, input_img, epoch=0):
        pred = outputs['output']
        
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        edge = self.edge_loss(pred, target)
        color = self.color_loss(pred, target, input_img)
        freq = self.freq_loss(pred, target)
        
        total_loss = (
            1.0 * l1 +
            0.5 * ssim +
            0.8 * perceptual +
            0.3 * edge +
            0.4 * color +
            0.2 * freq
        )
        
        return {
            'total': total_loss,
            'l1': l1,
            'ssim': ssim,
            'perceptual': perceptual,
            'edge': edge,
            'color': color,
            'freq': freq
        }


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=1000)
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('-l', '--load', type=str, default=None)
    parser.add_argument('-hor', '--hold_out_ratio', type=float, default=0.95)
    parser.add_argument('-s', '--image_size', type=int, default=286)
    parser.add_argument('-cs','--crop_size', type=int, default=256)
    parser.add_argument('-lr','--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use mixed precision training')
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


def check_dir():
    for d in ['./logs', './checkpoints', './result', './result/shadow_visualization']:
        os.makedirs(d, exist_ok=True)


def compute_psnr(pred, target):
    pred_u = (pred + 1.0) * 0.5
    target_u = (target + 1.0) * 0.5
    mse = F.mse_loss(pred_u, target_u)
    if mse.item() < 1e-10:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def compute_ssim_metric(pred, target):
    pred_u = (pred + 1.0) * 0.5
    target_u = (target + 1.0) * 0.5
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(pred_u, 3, 1, 1)
    mu2 = F.avg_pool2d(target_u, 3, 1, 1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(pred_u * pred_u, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(target_u * target_u, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(pred_u * target_u, 3, 1, 1) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def evaluate(model, val_dataset, device, save_path, epoch):
    model.eval()
    os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():
        num_imgs = min(8, len(val_dataset))
        grid_inputs = []
        grid_outputs = []
        grid_gts = []
        
        for i in range(num_imgs):
            img, _, gt_img = val_dataset[i]
            img = img.unsqueeze(0).to(device, non_blocking=True)
            gt_img = gt_img.unsqueeze(0).to(device, non_blocking=True)
            
            outputs = model(img)
            
            grid_inputs.append(unnormalize(img.cpu()))
            grid_outputs.append(unnormalize(outputs['output'].cpu()))
            grid_gts.append(unnormalize(gt_img.cpu()))
        
        grid_inputs = torch.cat(grid_inputs, dim=0)
        grid_outputs = torch.cat(grid_outputs, dim=0)
        grid_gts = torch.cat(grid_gts, dim=0)
        
        grid_all = torch.cat([grid_inputs, grid_gts, grid_outputs], dim=0)
        grid = make_grid(grid_all, nrow=num_imgs, padding=2, normalize=False)
        save_image(grid, f'{save_path}/epoch_{epoch}.png')


def visualize_shadow_detection(model, val_dataset, device, save_path, epoch):
    model.eval()
    os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():
        num_imgs = min(8, len(val_dataset))
        all_masks = []
        
        for i in range(num_imgs):
            img, _, gt_img = val_dataset[i]
            img = img.unsqueeze(0).to(device, non_blocking=True)
            gt_img = gt_img.unsqueeze(0).to(device, non_blocking=True)
            
            outputs = model(img)
            
            # Get shadow mask and uncertainty (already in [0, 1] from Sigmoid)
            mask = outputs['shadow_mask'].cpu().clamp(0, 1)
            uncertainty = outputs['uncertainty'].cpu().clamp(0, 1)
            
            # Append visualizations: Input | GT | Output | soft Mask | Uncertainty
            all_masks.append(unnormalize(img.cpu()))
            all_masks.append(unnormalize(gt_img.cpu()))
            all_masks.append(unnormalize(outputs['output'].cpu()))
            all_masks.append(mask.repeat(1, 3, 1, 1))  # Repeat to 3 channels for visualization
            all_masks.append(uncertainty.repeat(1, 3, 1, 1))  # Repeat to 3 channels
        
        all_masks = torch.cat(all_masks, dim=0)
        grid = make_grid(all_masks, nrow=5, padding=2, normalize=False)
        save_image(grid, f'{save_path}/epoch_{epoch}.png')


def plot_loss_curves(loss_history, save_name):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{save_name} Training Progress', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    if len(loss_history['total']) > 0:
        epochs = range(1, len(loss_history['total']) + 1)
        ax.plot(epochs, loss_history['total'], linewidth=2, color='#e74c3c')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Total Training Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    if len(loss_history['l1']) > 0:
        epochs = range(1, len(loss_history['l1']) + 1)
        ax.plot(epochs, loss_history['l1'], linewidth=2, color='#3498db', label='L1')
        ax.plot(epochs, loss_history['ssim'], linewidth=2, color='#2ecc71', label='SSIM')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('L1 + SSIM Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    ax = axes[0, 2]
    if len(loss_history['perceptual']) > 0:
        epochs = range(1, len(loss_history['perceptual']) + 1)
        ax.plot(epochs, loss_history['perceptual'], linewidth=2, color='#9b59b6', label='Perceptual')
        ax.plot(epochs, loss_history['edge'], linewidth=2, color='#e67e22', label='Edge')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Perceptual + Edge Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    ax = axes[1, 0]
    if len(loss_history['lr']) > 0:
        epochs = range(1, len(loss_history['lr']) + 1)
        ax.plot(epochs, loss_history['lr'], linewidth=2, color='#1abc9c')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    if len(loss_history['psnr']) > 0:
        psnr_epochs = [5*i for i in range(1, len(loss_history['psnr'])+1)]
        ax.plot(psnr_epochs, loss_history['psnr'], linewidth=2, color='#e74c3c',
                marker='o', markersize=6)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('PSNR (dB)', fontsize=11)
        ax.set_title('Validation PSNR', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if len(loss_history['psnr']) > 0:
            best_psnr = max(loss_history['psnr'])
            best_epoch = psnr_epochs[loss_history['psnr'].index(best_psnr)]
            ax.axhline(y=best_psnr, color='r', linestyle='--', alpha=0.5)
            ax.text(0.02, 0.98, f'Best: {best_psnr:.2f} dB @ Epoch {best_epoch}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax = axes[1, 2]
    if len(loss_history['ssim_metric']) > 0:
        ssim_epochs = [5*i for i in range(1, len(loss_history['ssim_metric'])+1)]
        ax.plot(ssim_epochs, loss_history['ssim_metric'], linewidth=2, color='#3498db',
                marker='s', markersize=6)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('SSIM', fontsize=11)
        ax.set_title('Validation SSIM', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if len(loss_history['ssim_metric']) > 0:
            best_ssim = max(loss_history['ssim_metric'])
            best_epoch = ssim_epochs[loss_history['ssim_metric'].index(best_ssim)]
            ax.axhline(y=best_ssim, color='r', linestyle='--', alpha=0.5)
            ax.text(0.02, 0.98, f'Best: {best_ssim:.4f} @ Epoch {best_epoch}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'./logs/{save_name}_loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def train_model(model, dataloader, val_dataset, num_epochs, parser, save_name='SAIFormer'):
    check_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    print(f"Batch Size: {parser.batch_size}")
    print(f"Initial LR: {parser.lr}")
    print(f"Mixed Precision (AMP): {parser.use_amp}")
    print(f"Workers: {parser.num_workers}")
    print(f"Hold-out: {parser.hold_out_ratio} ({len(dataloader.dataset)} train, {len(val_dataset)} val)")
    
    model = model.to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=parser.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        fused=False
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    
    criterion = EnhancedLoss(device=device)
    scaler = torch.cuda.amp.GradScaler() if parser.use_amp else None
    
    loss_history = {
        'total': [], 'l1': [], 'ssim': [], 'perceptual': [],
        'edge': [], 'color': [], 'lr': [], 'psnr': [], 'ssim_metric': []
    }
    
    best_psnr = 0.0
    nan_batch_count = 0  # Track total NaN batches across all epochs
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {
            'total': 0.0, 'l1': 0.0, 'ssim': 0.0,
            'perceptual': 0.0, 'edge': 0.0, 'color': 0.0
        }
        
        t_start = time.time()
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        num_batches = 0
        epoch_nan_count = 0  # Track NaN batches in current epoch
        
        for batch_idx, (shadow, _, gt) in enumerate(progress_bar):
            shadow = shadow.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            
            # NaN HANDLING 1: Input validation
            if torch.isnan(shadow).any() or torch.isnan(gt).any():
                print(f"WARNING: Skipping batch {batch_idx}: Input contains NaN")
                epoch_nan_count += 1
                continue
            
            optimizer.zero_grad(set_to_none=True)
            
            if parser.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(shadow)
                    
                    # NaN HANDLING 2: Output validation
                    if torch.isnan(outputs['output']).any():
                        print(f"WARNING: Skipping batch {batch_idx}: Model output contains NaN")
                        epoch_nan_count += 1
                        continue
                    
                    losses = criterion(outputs, gt, shadow, epoch)
                    
                    # NaN HANDLING 3: Loss validation
                    if torch.isnan(losses['total']) or torch.isinf(losses['total']):
                        print(f"CRITICAL: NaN/Inf loss at epoch {epoch+1}, batch {batch_idx}!")
                        print(f"   L1: {losses['l1'].item():.4f}, Perceptual: {losses['perceptual'].item():.4f}")
                        print(f"   Edge: {losses['edge'].item():.4f}, Color: {losses['color'].item():.4f}")
                        epoch_nan_count += 1
                        continue
                
                scaler.scale(losses['total']).backward()
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # NaN HANDLING 4: Gradient validation
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"WARNING: NaN/Inf gradient norm, skipping batch {batch_idx}")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    epoch_nan_count += 1
                    continue
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(shadow)
                
                # NaN HANDLING 2: Output validation
                if torch.isnan(outputs['output']).any():
                    print(f"WARNING: Skipping batch {batch_idx}: Model output contains NaN")
                    epoch_nan_count += 1
                    continue
                
                losses = criterion(outputs, gt, shadow, epoch)
                
                # NaN HANDLING 3: Loss validation
                if torch.isnan(losses['total']) or torch.isinf(losses['total']):
                    print(f"CRITICAL: NaN/Inf loss at epoch {epoch+1}, batch {batch_idx}!")
                    print(f"   L1: {losses['l1'].item():.4f}, Perceptual: {losses['perceptual'].item():.4f}")
                    print(f"   Edge: {losses['edge'].item():.4f}, Color: {losses['color'].item():.4f}")
                    epoch_nan_count += 1
                    continue
                
                losses['total'].backward()
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # NaN HANDLING 4: Gradient validation
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"WARNING: NaN/Inf gradient norm, skipping batch {batch_idx}")
                    optimizer.zero_grad(set_to_none=True)
                    epoch_nan_count += 1
                    continue
                
                optimizer.step()
            
            for key in epoch_losses.keys():
                epoch_losses[key] += losses[key].item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'L': f"{losses['total'].item():.3f}",
                'L1': f"{losses['l1'].item():.3f}",
                'P': f"{losses['perceptual'].item():.3f}"
            })
        
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches
            loss_history[key].append(epoch_losses[key])
        
        loss_history['lr'].append(optimizer.param_groups[0]['lr'])
        scheduler.step()
        
        epoch_time = time.time() - t_start
        
        # Report NaN batches if any were skipped
        if epoch_nan_count > 0:
            print(f"INFO: Skipped {epoch_nan_count} batches due to NaN/Inf in epoch {epoch+1}")
            nan_batch_count += epoch_nan_count
        
        # Safety check: if all batches were invalid, stop training
        if num_batches == 0:
            print(f"ERROR: All batches in epoch {epoch+1} were invalid! Stopping training.")
            break
        
        if (epoch + 1) == 1 or (epoch + 1) % 5 == 0:
            model.eval()
            val_psnr = 0.0
            val_ssim = 0.0
            num_val = min(30, len(val_dataset))
            
            with torch.no_grad():
                for i in range(num_val):
                    img, _, gt_img = val_dataset[i]
                    img = img.unsqueeze(0).to(device, non_blocking=True)
                    gt_img = gt_img.unsqueeze(0).to(device, non_blocking=True)
                    
                    if parser.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(img)
                    else:
                        outputs = model(img)
                    
                    val_psnr += compute_psnr(outputs['output'], gt_img)
                    val_ssim += compute_ssim_metric(outputs['output'], gt_img)
            
            val_psnr /= num_val
            val_ssim /= num_val
            loss_history['psnr'].append(val_psnr)
            loss_history['ssim_metric'].append(val_ssim)
            model.train()
            
            print(f"\n[Epoch {epoch+1}] Time: {epoch_time:.1f}s ({epoch_time/60:.2f}min)")
            print(f"  Total: {epoch_losses['total']:.4f} | L1: {epoch_losses['l1']:.4f} | SSIM: {epoch_losses['ssim']:.4f}")
            print(f"  Perceptual: {epoch_losses['perceptual']:.4f} | Edge: {epoch_losses['edge']:.4f}")
            print(f"  Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.7f}")
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_psnr': best_psnr,
                    'val_ssim': val_ssim
                }
                torch.save(checkpoint, f'checkpoints/{save_name}_best.pth')
                print(f"  Best model saved! PSNR: {best_psnr:.2f} dB")
        else:
            print(f"\n[Epoch {epoch+1}] Time: {epoch_time:.1f}s | "
                  f"Total: {epoch_losses['total']:.4f}, "
                  f"L1: {epoch_losses['l1']:.4f}, "
                  f"SSIM: {epoch_losses['ssim']:.4f}, "
                  f"Perceptual: {epoch_losses['perceptual']:.4f}, "
                  f"Edge: {epoch_losses['edge']:.4f}")
        
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'loss_history': loss_history
            }
            torch.save(checkpoint, f'checkpoints/{save_name}_{epoch+1}.pth')
            print(f"  Checkpoint saved: {save_name}_{epoch+1}.pth")
        
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            evaluate(model, val_dataset, device, './result', epoch+1)
            visualize_shadow_detection(model, val_dataset, device, './result/shadow_visualization', epoch+1)
            print(f"  Visual grids saved: epoch {epoch+1}")
        
        if (epoch + 1) % 5 == 0 or (epoch + 1) == 1:
            plot_loss_curves(loss_history, save_name)
            print("  Loss curves saved")
    
    print("\nTraining Complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Total NaN batches skipped: {nan_batch_count}")
    
    return model


def main(parser):
    model = SAGNet(
        input_channels=3,
        output_channels=3,
        embed_dim=64,
        num_blocks=[2, 2, 2, 2],
        num_heads=[2, 4, 8, 16]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    if parser.load:
        checkpoint_path = f'./checkpoints/{parser.load}.pth'
        print(f'Loading: {checkpoint_path}')
        try:
            model.load_state_dict(fix_model_state_dict(torch.load(checkpoint_path)))
            print("Loaded!\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    train_img_list, val_img_list = make_datapath_list(phase='train', rate=parser.hold_out_ratio)
    
    mean = (0.5,)
    std = (0.5,)
    
    train_dataset = ImageDataset(
        img_list=train_img_list,
        img_transform=ImageTransform(size=parser.image_size, crop_size=parser.crop_size, mean=mean, std=std),
        phase='train'
    )
    
    val_dataset = ImageDataset(
        img_list=val_img_list,
        img_transform=ImageTransform(size=parser.image_size, crop_size=parser.crop_size, mean=mean, std=std),
        phase='test_no_crop'
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=parser.batch_size,
        shuffle=True,
        num_workers=parser.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    train_model(model, train_dataloader, val_dataset, parser.epoch, parser, save_name='SAGNet')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)