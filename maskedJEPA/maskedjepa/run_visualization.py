#!/usr/bin/env python3
"""
Standalone visualization script for adaptive masking.
Run this to generate visualization videos without waiting for training to complete.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import cv2
from PIL import Image
import os
import sys

# Add the src/masks directory to the path
sys.path.append('src/masks')
from adaptive import AdaptiveMaskCollator

def create_dummy_video(batch_size=2, num_frames=8, height=192, width=192, channels=3):
    """Create a dummy video tensor for visualization"""
    # Create a video with some patterns to make masking interesting
    video = torch.randn(batch_size, channels, num_frames, height, width)
    
    # Add some patterns to make it more interesting
    for b in range(batch_size):
        for t in range(num_frames):
            # Add a moving circle
            center_x = int(width * 0.3 + t * 10)
            center_y = int(height * 0.5)
            radius = 20
            
            for i in range(max(0, center_y-radius), min(height, center_y+radius)):
                for j in range(max(0, center_x-radius), min(width, center_x+radius)):
                    if (i - center_y)**2 + (j - center_x)**2 <= radius**2:
                        video[b, :, t, i, j] = torch.tensor([1.0, 0.5, 0.2])  # Red circle
            
            # Add a static rectangle
            rect_x1, rect_y1 = 50, 50
            rect_x2, rect_y2 = 100, 100
            # video[b, :, t, rect_y1:rect_y2, rect_x1:rect_x2] = torch.tensor([0.2, 0.8, 1.0])  # Blue rectangle
            video[b, :, t, rect_y1:rect_y2, rect_x1:rect_x2] = torch.tensor([0.2, 0.8, 1.0]).view(3, 1, 1)
    
    return video

def patch_embed_fn_dummy(video):
    """Dummy patch embedding function for visualization"""
    # Simulate patch embedding: [B, C, T, H, W] -> [B, N, D]
    B, C, T, H, W = video.shape
    patch_size = 16
    
    # Calculate number of patches
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches_t = T // 1  # Assuming tubelet_size = 1
    total_patches = num_patches_h * num_patches_w * num_patches_t
    
    # Simulate patch embedding
    # In reality, this would be the actual patch embedding from the model
    patch_embeddings = torch.randn(B, total_patches, 1280)  # 1280 for ViT-Huge
    
    return patch_embeddings

def visualize_patch_importance_video(importance_scores, num_patches_h, num_patches_w, num_patches_t, 
                                   sample_idx=0, save_path="masking_visualizations/patch_importance_video.gif"):
    """Create a video showing patch importance evolution over time"""
    import imageio
    patches_per_frame = num_patches_h * num_patches_w
    num_frames = num_patches_t
    frames = []
    
    for frame_idx in range(num_frames):
        # Extract importance scores for this frame
        frame_start_idx = frame_idx * patches_per_frame
        frame_end_idx = frame_start_idx + patches_per_frame
        frame_importance = importance_scores[sample_idx][frame_start_idx:frame_end_idx]
        importance_2d = frame_importance.reshape(num_patches_h, num_patches_w)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Heatmap of importance scores
        im1 = ax1.imshow(importance_2d.cpu().numpy(), cmap='viridis', interpolation='nearest')
        ax1.set_title(f'Patch Importance Scores (Frame {frame_idx})')
        ax1.set_xlabel('Patch X')
        ax1.set_ylabel('Patch Y')
        plt.colorbar(im1, ax=ax1)
        
        # Top-k important patches highlighted
        k = 20  # Show top 20 most important patches
        top_k_indices = torch.topk(importance_scores[sample_idx][frame_start_idx:frame_end_idx], k).indices
        
        # Create binary mask for top-k patches
        top_k_mask = torch.zeros_like(frame_importance)
        top_k_mask[top_k_indices] = 1
        top_k_mask_2d = top_k_mask.reshape(num_patches_h, num_patches_w)
        
        im2 = ax2.imshow(top_k_mask_2d.cpu().numpy(), cmap='Reds', interpolation='nearest')
        ax2.set_title(f'Top-{k} Most Important Patches (Frame {frame_idx})')
        ax2.set_xlabel('Patch X')
        ax2.set_ylabel('Patch Y')
        
        plt.tight_layout()
        
        # Save to buffer
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(save_path, frames, duration=0.8)
    print(f"Patch importance video saved to {save_path}")

def visualize_masks_video(masks_enc, masks_pred, num_patches_h, num_patches_w, num_patches_t,
                         sample_idx=0, save_path="masking_visualizations/mask_visualization_video.gif"):
    """Create a video showing mask evolution over time"""
    import imageio
    patches_per_frame = num_patches_h * num_patches_w
    num_frames = num_patches_t
    total_patches = num_patches_h * num_patches_w * num_patches_t
    frames = []
    
    for frame_idx in range(num_frames):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for mask_idx in range(min(2, len(masks_enc))):  # Show first 2 masks
            # Get masks for this sample
            if isinstance(masks_enc[mask_idx], torch.Tensor):
                mask_e = masks_enc[mask_idx][sample_idx]  # Masked indices
                mask_p = masks_pred[mask_idx][sample_idx]  # Visible indices
            else:
                mask_e = masks_enc[mask_idx][0][sample_idx]
                mask_p = masks_pred[mask_idx][0][sample_idx]
            
            # Create binary masks for this frame
            frame_start_idx = frame_idx * patches_per_frame
            frame_end_idx = frame_start_idx + patches_per_frame
            
            masked_mask = torch.zeros(patches_per_frame, dtype=torch.bool)
            visible_mask = torch.zeros(patches_per_frame, dtype=torch.bool)
            
            # Filter masks for this frame
            for patch_idx in mask_e:
                if frame_start_idx <= patch_idx < frame_end_idx:
                    local_idx = patch_idx - frame_start_idx
                    masked_mask[local_idx] = True
            
            for patch_idx in mask_p:
                if frame_start_idx <= patch_idx < frame_end_idx:
                    local_idx = patch_idx - frame_start_idx
                    visible_mask[local_idx] = True
            
            # Reshape to spatial dimensions
            masked_2d = masked_mask.reshape(num_patches_h, num_patches_w)
            visible_2d = visible_mask.reshape(num_patches_h, num_patches_w)
            
            # Plot masked patches
            axes[0, mask_idx].imshow(masked_2d.cpu().numpy(), cmap='Reds', interpolation='nearest')
            axes[0, mask_idx].set_title(f'Mask {mask_idx+1}: Masked Patches (Frame {frame_idx})')
            axes[0, mask_idx].set_xlabel('Patch X')
            axes[0, mask_idx].set_ylabel('Patch Y')
            
            # Plot visible patches
            axes[1, mask_idx].imshow(visible_2d.cpu().numpy(), cmap='Greens', interpolation='nearest')
            axes[1, mask_idx].set_title(f'Mask {mask_idx+1}: Visible Patches (Frame {frame_idx})')
            axes[1, mask_idx].set_xlabel('Patch X')
            axes[1, mask_idx].set_ylabel('Patch Y')
        
        plt.tight_layout()
        
        # Save to buffer
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(save_path, frames, duration=0.8)
    print(f"Mask visualization video saved to {save_path}")

def create_comprehensive_masking_video(video, importance_scores, masks_enc, masks_pred, 
                                     num_patches_h, num_patches_w, sample_idx=0, 
                                     save_path="masking_visualizations/comprehensive_masking_video.gif"):
    """Create a comprehensive video showing all aspects of adaptive masking"""
    import imageio
    patch_size = 16
    num_frames = video.shape[2]
    patches_per_frame = num_patches_h * num_patches_w
    frames = []
    
    # Get visible patches for this sample
    if isinstance(masks_pred[0], torch.Tensor):
        visible_indices = masks_pred[0][sample_idx]
    else:
        visible_indices = masks_pred[0][0][sample_idx]
    
    for frame_idx in range(num_frames):
        # Get the video frame
        frame = video[sample_idx, :, frame_idx].permute(1, 2, 0)  # [H, W, C]
        frame_np = frame.cpu().numpy()
        frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min())
        
        # Importance scores for this frame
        frame_start_idx = frame_idx * patches_per_frame
        frame_end_idx = frame_start_idx + patches_per_frame
        frame_importance = importance_scores[sample_idx][frame_start_idx:frame_end_idx]
        importance_2d = frame_importance.reshape(num_patches_h, num_patches_w)
        
        # Create mask overlay for this frame
        mask_overlay = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        for patch_idx in visible_indices:
            if frame_start_idx <= patch_idx < frame_end_idx:
                local_patch_idx = patch_idx - frame_start_idx
                patch_h = (local_patch_idx // num_patches_w) % num_patches_h
                patch_w = local_patch_idx % num_patches_w
                
                pixel_h_start = patch_h * patch_size
                pixel_h_end = pixel_h_start + patch_size
                pixel_w_start = patch_w * patch_size
                pixel_w_end = pixel_w_start + patch_size
                
                mask_overlay[pixel_h_start:pixel_h_end, pixel_w_start:pixel_w_end] = 1.0
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Row 1: Original, Importance, Combined
        axes[0, 0].imshow(frame_np)
        axes[0, 0].set_title(f'Original Frame {frame_idx}')
        axes[0, 0].axis('off')
        
        im1 = axes[0, 1].imshow(importance_2d.cpu().numpy(), cmap='viridis', interpolation='nearest')
        axes[0, 1].set_title(f'Patch Importance (Frame {frame_idx})')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        axes[0, 2].imshow(frame_np)
        axes[0, 2].imshow(mask_overlay, cmap='Greens', alpha=0.6)
        axes[0, 2].set_title(f'Visible Patches (Frame {frame_idx})')
        axes[0, 2].axis('off')
        
        # Row 2: Statistics and detailed views
        # Mask statistics
        visible_count = np.sum(mask_overlay > 0) / (patch_size * patch_size)
        masked_count = patches_per_frame - visible_count
        
        stats_text = f"""Frame {frame_idx} Statistics:
        Total patches: {patches_per_frame}
        Visible patches: {int(visible_count)}
        Masked patches: {int(masked_count)}
        Visible ratio: {visible_count/patches_per_frame:.2%}
        """
        
        axes[1, 0].text(0.1, 0.5, stats_text, transform=axes[1, 0].transAxes, 
                        fontsize=10, verticalalignment='center')
        axes[1, 0].set_title('Masking Statistics')
        axes[1, 0].axis('off')
        
        # Top important patches
        k = 10
        top_k_indices = torch.topk(frame_importance, k).indices
        top_k_mask = torch.zeros_like(frame_importance)
        top_k_mask[top_k_indices] = 1
        top_k_mask_2d = top_k_mask.reshape(num_patches_h, num_patches_w)
        
        axes[1, 1].imshow(top_k_mask_2d.cpu().numpy(), cmap='Reds', interpolation='nearest')
        axes[1, 1].set_title(f'Top-{k} Important Patches')
        axes[1, 1].axis('off')
        
        # Importance + masking combined
        axes[1, 2].imshow(importance_2d.cpu().numpy(), cmap='viridis', alpha=0.7)
        # axes[1, 2].imshow(mask_overlay.reshape(num_patches_h, num_patches_w), 
        #                   cmap='Greens', alpha=0.5)
        # Create a patch-level mask for the current frame
        patch_mask = np.zeros((num_patches_h, num_patches_w), dtype=np.float32)
        for patch_idx in visible_indices:
            if frame_start_idx <= patch_idx < frame_end_idx:
                local_patch_idx = patch_idx - frame_start_idx
                patch_h = (local_patch_idx // num_patches_w) % num_patches_h
                patch_w = local_patch_idx % num_patches_w
                patch_mask[patch_h, patch_w] = 1.0

        axes[1, 2].imshow(importance_2d.cpu().numpy(), cmap='viridis', alpha=0.7)
        axes[1, 2].imshow(patch_mask, cmap='Greens', alpha=0.5)
        axes[1, 2].set_title('Importance + Masking')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save to buffer
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(save_path, frames, duration=1.0)
    print(f"Comprehensive masking video saved to {save_path}")

def main():
    """Main visualization function"""
    print("🎬 Starting Standalone Adaptive Masking Visualization...")
    
    # Create output directory
    os.makedirs("masking_visualizations", exist_ok=True)
    
    # Parameters
    batch_size = 2
    num_frames = 8
    height, width = 192, 192
    channels = 3
    patch_size = 16
    tubelet_size = 1
    
    # Calculate patch dimensions
    num_patches_h = height // patch_size  # 12
    num_patches_w = width // patch_size   # 12
    num_patches_t = num_frames // tubelet_size  # 8
    total_patches = num_patches_h * num_patches_w * num_patches_t  # 1152
    
    print(f"📐 Patch dimensions: {num_patches_h}x{num_patches_w}x{num_patches_t} = {total_patches} total patches")
    
    # Create dummy video
    print("🎥 Creating dummy video...")
    video = create_dummy_video(batch_size, num_frames, height, width, channels)
    print(f"Video shape: {video.shape}")
    
    # Create adaptive mask collator
    print("🎭 Initializing adaptive mask collator...")
    mask_collator = AdaptiveMaskCollator(
        num_tokens=total_patches,
        visible_patches=75,  # Keep 75 patches visible
        patch_embed_fn=patch_embed_fn_dummy
    )
    
    # Create dummy batch
    batch = []
    for i in range(batch_size):
        # Simulate the batch format expected by the collator
        # batch.append(([video[i]], None, [i]))  # (video_clips, label, indices)
        batch.append(([video[i]], 0, [i]))
    # Generate masks
    print("🎯 Generating adaptive masks...")
    collated_batch, masks_enc_list, masks_pred_list = mask_collator(batch)
    
    print(f"Generated {len(masks_enc_list)} mask strategies")
    print(f"Mask shapes: enc={[m.shape for m in masks_enc_list]}, pred={[m.shape for m in masks_pred_list]}")
    
    # Create comprehensive video visualizations
    print("📊 Creating patch importance video...")
    visualize_patch_importance_video(
        mask_collator.last_importance_scores, num_patches_h, num_patches_w, num_patches_t,
        sample_idx=0, save_path="masking_visualizations/patch_importance_video.gif"
    )
    
    print("🎭 Creating mask visualization video...")
    visualize_masks_video(
        masks_enc_list, masks_pred_list, num_patches_h, num_patches_w, num_patches_t,
        sample_idx=0, save_path="masking_visualizations/mask_visualization_video.gif"
    )
    
    print("🎞️ Creating comprehensive masking video...")
    create_comprehensive_masking_video(
        video, mask_collator.last_importance_scores, masks_enc_list, masks_pred_list,
        num_patches_h, num_patches_w, sample_idx=0,
        save_path="masking_visualizations/comprehensive_masking_video.gif"
    )
    
    # Print statistics
    print("\n📈 Masking Statistics:")
    for i, (masks_enc, masks_pred) in enumerate(zip(masks_enc_list, masks_pred_list)):
        if isinstance(masks_enc, torch.Tensor):
            masked_count = masks_enc[0].shape[0]
            visible_count = masks_pred[0].shape[0]
        else:
            masked_count = masks_enc[0][0].shape[0]
            visible_count = masks_pred[0][0].shape[0]
        
        print(f"  Mask {i+1}: {visible_count} visible, {masked_count} masked")
        print(f"  Mask ratio: {visible_count/total_patches:.2%} visible")
    
    print("\n✅ Visualization complete! Check the 'masking_visualizations' folder.")
    print("📁 Generated video files:")
    print("  - patch_importance_video.gif")
    print("  - mask_visualization_video.gif")
    print("  - comprehensive_masking_video.gif")
    print(f"📂 Files saved in: {os.path.abspath('masking_visualizations')}")

if __name__ == "__main__":
    main() 