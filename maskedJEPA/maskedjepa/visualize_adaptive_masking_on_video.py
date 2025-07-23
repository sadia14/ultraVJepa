import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import imageio
from src.masks.adaptive import AdaptiveMaskCollator

def load_video_as_tensor(filename, num_frames=8, height=192, width=192):
    cap = cv2.VideoCapture(filename)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    frames = np.stack(frames, axis=0)  # [T, H, W, C]
    frames = frames.transpose(3, 0, 1, 2)  # [C, T, H, W]
    video_tensor = torch.from_numpy(frames).float() / 255.0
    return video_tensor  # [C, T, H, W]

def visualize_frame(video, importance_scores, masks_pred, num_patches_h, num_patches_w, frame_idx, sample_idx=0):
    frame = video[sample_idx, :, frame_idx].permute(1, 2, 0)
    frame_np = frame.cpu().numpy()
    frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min())
    patch_size = 16
    patches_per_frame = num_patches_h * num_patches_w
    frame_start_idx = frame_idx * patches_per_frame
    frame_end_idx = frame_start_idx + patches_per_frame
    frame_importance = importance_scores[sample_idx][frame_start_idx:frame_end_idx]
    importance_2d = frame_importance.reshape(num_patches_h, num_patches_w)
    if isinstance(masks_pred[0], torch.Tensor):
        visible_indices = masks_pred[0][sample_idx]
    else:
        visible_indices = masks_pred[0][0][sample_idx]
    mask_overlay = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    patch_mask = np.zeros((num_patches_h, num_patches_w), dtype=np.float32)
    for patch_idx in visible_indices:
        if frame_start_idx <= patch_idx < frame_end_idx:
            local_patch_idx = patch_idx - frame_start_idx
            patch_h = (local_patch_idx // num_patches_w) % num_patches_h
            patch_w = local_patch_idx % num_patches_w
            patch_mask[patch_h, patch_w] = 1.0
            pixel_h_start = patch_h * patch_size
            pixel_h_end = pixel_h_start + patch_size
            pixel_w_start = patch_w * patch_size
            pixel_w_end = pixel_w_start + patch_size
            mask_overlay[pixel_h_start:pixel_h_end, pixel_w_start:pixel_w_end] = 1.0
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0, 0].imshow(frame_np)
    axes[0, 0].set_title('Original Video Frame')
    axes[0, 0].axis('off')
    im1 = axes[0, 1].imshow(importance_2d.cpu().numpy(), cmap='viridis', interpolation='nearest')
    axes[0, 1].set_title('Patch Importance Scores')
    axes[0, 1].set_xlabel('Patch X')
    axes[0, 1].set_ylabel('Patch Y')
    plt.colorbar(im1, ax=axes[0, 1])
    axes[1, 0].imshow(frame_np)
    axes[1, 0].imshow(mask_overlay, cmap='Greens', alpha=0.6)
    axes[1, 0].set_title('Video with Visible Patches')
    axes[1, 0].axis('off')
    total_patches = num_patches_h * num_patches_w
    visible_count = int(np.sum(patch_mask))
    masked_count = total_patches - visible_count
    stats_text = f"""Masking Statistics:\nTotal patches: {total_patches}\nVisible patches: {visible_count}\nMasked patches: {masked_count}\nVisible ratio: {visible_count/total_patches:.2%}"""
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, fontsize=12, verticalalignment='center')
    axes[1, 1].set_title('Masking Statistics')
    axes[1, 1].axis('off')
    plt.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

def main():
    # --- USER: Set your video path and model checkpoint here ---
    video_path = '/home/ext_kamal_sadia2_mayo_edu/jepa/dataset/videos/CmNo7kb8sTY.mp4'  # <-- CHANGE THIS
    checkpoint_path = '/home/ext_kamal_sadia2_mayo_edu/jepa/logs/jepa-latest.pth.tar'  # <-- CHANGE THIS
    # ----------------------------------------------------------

    num_frames = 8
    height, width = 192, 192
    patch_size = 16
    tubelet_size = 1
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_patches_t = num_frames // tubelet_size
    total_patches = num_patches_h * num_patches_w * num_patches_t
    os.makedirs('vjepa_masking_gifs', exist_ok=True)

    # Load video
    video = load_video_as_tensor(video_path, num_frames, height, width).unsqueeze(0)  # [1, C, T, H, W]

    # --- Load your trained model ---
    # Replace this with your actual model class and loading code
    from app.vjepa.train import init_video_model

    encoder, predictor = init_video_model(
        uniform_power=True,
        use_mask_tokens=True,
        num_mask_tokens=1,
        zero_init_mask_tokens=True,
        device='cuda',
        patch_size=16,
        num_frames=8,
        tubelet_size=1,
        model_name='vit_huge',
        crop_size=192,
        pred_depth=4,
        pred_embed_dim=384,
        use_sdpa=False
    )
    # checkpoint = torch.load(checkpoint_path, map_location='cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder.load_state_dict(checkpoint['encoder'])        
    encoder.eval()
    model = encoder

    # After loading the model and before uing the video tensor
    device = next(model.parameters()).device
    video = video.to(device)

    # Patch embedding function using your trained model
    def patch_embed_fn(video):
        with torch.no_grad():
            return model.backbone.patch_embed(video)

    # AdaptiveMaskCollator using your trained model's patch embedding
    mask_collator = AdaptiveMaskCollator(
        num_tokens=total_patches,
        visible_ratio= 0.25,
        patch_embed_fn=patch_embed_fn
    )

    # Prepare batch for collator
    batch = [([video[0]], 0, [0])]
    collated_batch, masks_enc_list, masks_pred_list = mask_collator(batch)

    frames = []
    for frame_idx in range(num_frames):
        img = visualize_frame(
            video, mask_collator.last_importance_scores, masks_pred_list,
            num_patches_h, num_patches_w, frame_idx, sample_idx=0
        
        )
        print("Importance scores video 1:", mask_collator.last_importance_scores[0][:10])
        frames.append(img)

    gif_path = f'vjepa_masking_gifs/adaptive_masking_on_video_trained_baby.gif'
    imageio.mimsave(gif_path, frames, duration=0.7)
    print(f'🎞️ GIF saved to {gif_path}')

if __name__ == "__main__":
    main() 

