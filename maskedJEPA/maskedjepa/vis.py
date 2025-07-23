# visualize_adaptive_masking.py

import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.masks.adaptive import AdaptiveMaskCollator
from app.vjepa.utils import init_video_model, load_checkpoint
from app.vjepa.transforms import make_transforms
from src.datasets.data_manager import init_data


def visualize_frames(video, masks_enc, step, save_dir="masking_debug"):
    """
    Save per-frame visualizations of masked regions in a video sample.
    video: [B, C, T, H, W]
    masks_enc: [B, N], binary mask (1=visible, 0=masked)
    """
    os.makedirs(save_dir, exist_ok=True)
    B, C, T, H, W = video.shape
    grid_h = grid_w = int((masks_enc.shape[1]) ** 0.5)  # assume square

    for i in range(min(B, 4)):  # visualize first 4 samples
        for t in range(T):
            fig, ax = plt.subplots()
            frame = video[i, :, t].permute(1, 2, 0).cpu().numpy()
            ax.imshow(frame)
            ax.set_title(f"Sample {i} Frame {t} (Step {step})")

            # Draw visible patch boundaries
            mask = masks_enc[i].reshape(grid_h, grid_w).cpu()
            patch_h, patch_w = H // grid_h, W // grid_w

            for y in range(grid_h):
                for x in range(grid_w):
                    if mask[y, x] == 1:
                        rect = plt.Rectangle((x * patch_w, y * patch_h), patch_w, patch_h,
                                             edgecolor='lime', facecolor='none', linewidth=1.5)
                        ax.add_patch(rect)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/sample{i}_frame{t}_step{step}.png")
            plt.close()


def run_visualization(checkpoint_path, cfg_model, cfg_data):
    # Step 1: Init model
    encoder, patch_embed_fn = init_video_model(cfg_model)
    
    encoder.eval()

    # Step 2: Init mask collator
    total_tokens = cfg_model['patches'] ** 2
    mask_collator = AdaptiveMaskCollator(
        num_tokens=total_tokens,
        visible_ratio=cfg_model.get('visible_ratio', 0.25),
        patch_embed_fn=patch_embed_fn,
        visualize=False  # we will call visualize manually
    )

    # Step 3: Load weights
    encoder, *_ = load_checkpoint(checkpoint_path, encoder=encoder,
                                  predictor=None, target_encoder=None)

    # Step 4: Init data loader
    transform = make_transforms(is_train=False)
    loader, _ = init_data(
        cfg_data,
        is_train=False,
        transform=transform,
        collator=mask_collator
    )

    # Step 5: Inference & visualization
    with torch.no_grad():
        for step, batch in tqdm(enumerate(loader), total=1):
            video = batch['video'].to('cuda')  # [B, C, T, H, W]
            _, masks_enc_list, _ = mask_collator(batch)
            masks_enc = masks_enc_list[0]  # assume 1 mask type
            visualize_frames(video, masks_enc, step)
            break  # remove this if you want to process more batches


if __name__ == "__main__":
    checkpoint_path = "/home/ext_kamal_sadia2_mayo_edu/jepa/logs/jepa-latest.pth.tar"

    # Configuration used during training
    cfg_model = {
        'model_name': 'pretrain_videomae_base_patch16_224',
        'patches': 14,  # e.g., for 224x224 images with patch_size=16 → 14x14 = 196
        'visible_ratio': 0.25,
        'device': 'cuda'
    }

    cfg_data = {
        'dataset': 'VideoDataset',
        'video_path': '/path/to/your/videos/',
        'csv_path': '/path/to/split/test.csv',
        'num_frames': 8,
        'sampling_rate': 2,
        'batch_size': 4,
        'num_workers': 2,
        'pin_memory': True
    }

    run_visualization(checkpoint_path, cfg_model, cfg_data)
