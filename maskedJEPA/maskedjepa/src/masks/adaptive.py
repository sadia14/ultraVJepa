import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from multiprocessing import Value
import matplotlib.pyplot as plt
import os


class AdaptiveMaskCollator(nn.Module):
    """
    AdaMAE-style masking: uses softmax-based importance sampling
    to select visible patches based on learned token importance.
    """
    def __init__(self, num_tokens, patch_embed_fn, visible_ratio=0.25, visualize=False):
        super().__init__()
        self.num_tokens = num_tokens
        # self.visible_patches = visible_patches
        self.visible_ratio = visible_ratio
        self.patch_embed_fn = patch_embed_fn  # A function that returns [B, N, D] from video
        self.visualize = visualize

        # We'll initialize components dynamically when we first see the data
        self.pos_embed_probs = None
        self.model_dim = None
        self.initialized = False

        # Lightweight importance estimator (will be initialized dynamically)
        self.token_attn = None
        self.linear = None
        self.softmax = nn.Softmax(dim=-1)

        # Counter for tracking iterations (like in reference)
        self._itr_counter = Value('i', -1)
        
        # Store last importance scores for visualization
        self.last_importance_scores = None
        self.last_masks = None
        self.visible_patches = int(num_tokens * visible_ratio)
        self.last_visible_patches = self.visible_patches 

    def _initialize_components(self, model_dim, device):
        """Dynamically initialize components based on the actual model dimension"""
        self.model_dim = model_dim
        
        # Initialize positional embeddings directly on the correct device
        self.pos_embed_probs = nn.Parameter(torch.randn(1, self.num_tokens, model_dim, device=device) * 0.02)
        
        # Initialize importance estimator components
        self.token_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=8,
            batch_first=True
        )
        self.linear = nn.Linear(model_dim, self.num_tokens)
        self.softmax = nn.Softmax(dim=-1)
        
        # Move all components to the same device as the input
        self.token_attn = self.token_attn.to(device)
        self.linear = self.linear.to(device)
        self.softmax = self.softmax.to(device)
        
        self.initialized = True

    def get_token_probs(self, x):
        """
        Run self-attention and project to importance logits.
        Args:
            x: [B, N, D] input tokens
        Returns:
            logits: [B, N]
        """
        z, _ = self.token_attn(x, x, x)
        logits = self.linear(z.mean(dim=1))  # [B, N]
        return logits

    def step(self):
        """ JEPA expects .step() to exist; here it's a no-op """
        pass

    def visualize_masking(self, video, importance_scores, masks_enc, masks_pred, 
                         sample_idx=0, frame_idx=0, save_dir="masking_debug"):
        """Visualize the masking process for debugging"""
        if not self.visualize:
            return
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Get video frame
        frame = video[sample_idx, :, frame_idx].permute(1, 2, 0)  # [H, W, C]
        frame_np = frame.cpu().numpy()
        frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min())
        
        # Calculate patch dimensions
        patch_size = 16
        num_patches_h = frame.shape[0] // patch_size  # 12
        num_patches_w = frame.shape[1] // patch_size  # 12
        num_frames = video.shape[2]  # 8
        num_patches_t = num_frames  # Assuming tubelet_size = 1
        
        # Calculate total patches per frame
        patches_per_frame = num_patches_h * num_patches_w  # 144
        
        # Extract importance scores for the specific frame
        frame_start_idx = frame_idx * patches_per_frame
        frame_end_idx = frame_start_idx + patches_per_frame
        frame_importance = importance_scores[sample_idx][frame_start_idx:frame_end_idx]
        
        # Reshape importance scores to spatial dimensions for this frame
        importance_2d = frame_importance.reshape(num_patches_h, num_patches_w)
        
        # Get visible patches
        if isinstance(masks_pred[0], torch.Tensor):
            visible_indices = masks_pred[0][sample_idx]
        else:
            visible_indices = masks_pred[0][0][sample_idx]
        
        # Create mask overlay for this specific frame
        mask_overlay = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        
        # Filter visible patches that belong to this frame
        frame_start_idx = frame_idx * patches_per_frame
        frame_end_idx = frame_start_idx + patches_per_frame
        
        for patch_idx in visible_indices:
            # Check if this patch belongs to the current frame
            if frame_start_idx <= patch_idx < frame_end_idx:
                # Convert global patch index to frame-local index
                local_patch_idx = patch_idx - frame_start_idx
                
                # Convert to spatial coordinates within the frame
                patch_h = (local_patch_idx // num_patches_w) % num_patches_h
                patch_w = local_patch_idx % num_patches_w
                
                pixel_h_start = patch_h * patch_size
                pixel_h_end = pixel_h_start + patch_size
                pixel_w_start = patch_w * patch_size
                pixel_w_end = pixel_w_start + patch_size
                
                mask_overlay[pixel_h_start:pixel_h_end, pixel_w_start:pixel_w_end] = 1.0
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original frame
        axes[0, 0].imshow(frame_np)
        axes[0, 0].set_title('Original Video Frame')
        axes[0, 0].axis('off')
        
        # Importance scores heatmap
        im1 = axes[0, 1].imshow(importance_2d.cpu().numpy(), cmap='viridis', interpolation='nearest')
        axes[0, 1].set_title('Patch Importance Scores')
        axes[0, 1].set_xlabel('Patch X')
        axes[0, 1].set_ylabel('Patch Y')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Visible patches overlay
        axes[1, 0].imshow(frame_np)
        axes[1, 0].imshow(mask_overlay, cmap='Greens', alpha=0.6)
        axes[1, 0].set_title('Video with Visible Patches')
        axes[1, 0].axis('off')
        
        # Mask statistics
        total_patches = num_patches_h * num_patches_w
        visible_count = len(visible_indices)
        masked_count = total_patches - visible_count
        
        stats_text = f"""Masking Statistics:
        Total patches: {total_patches}
        Visible patches: {visible_count}
        Masked patches: {masked_count}
        Visible ratio: {visible_count/total_patches:.2%}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Masking Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/masking_visualization_step_{self._itr_counter.value}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # print(f"🎭 Masking visualization saved to {save_dir}/masking_visualization_step_{self._itr_counter.value}.png")

    def get_mask_log_probs(self):
        # Return the log-probabilities of the selected tokens for the current batch
        return getattr(self, 'last_mask_log_probs', None)

    def get_mask_l_r(self):
        # Return the per-token reconstruction error for the current batch
        return getattr(self, 'last_mask_l_r', None)

    def get_bool_masked_pos(self):
        # Return the boolean mask for masked positions for the current batch
        return getattr(self, 'last_bool_masked_pos', None)

    def __call__(self, batch):
        """
        Args:
            batch: List of tuples like (video_clips_list, label, clip_indices)
        Returns:
            batch, masks_enc, masks_pred
        """
        batch_size = len(batch)
        
        # Extract video clips from each sample
        try:
            video_clips_lists = [x[0] if isinstance(x, (list, tuple)) else x for x in batch]
        except Exception as e:
            # print(f"[ERROR] Failed extracting video from batch: {e}")
            raise

        # Convert list of clips to tensor format
        processed_video = []
        for i, clips_list in enumerate(video_clips_lists):
            try:
                if isinstance(clips_list, list) and len(clips_list) > 0:
                    # Convert list of clips to tensor
                    clips_tensors = []
                    for j, clip in enumerate(clips_list):
                        if isinstance(clip, torch.Tensor):
                            clips_tensors.append(clip)
                        elif isinstance(clip, list):
                            # Convert nested list to tensor
                            try:
                                clip_tensor = torch.as_tensor(clip)
                                clips_tensors.append(clip_tensor)
                            except Exception as clip_error:
                                # print(f"[ERROR] Failed converting clip {j} to tensor: {clip_error}")
                                # Try numpy conversion as fallback
                                try:
                                    import numpy as np
                                    np_array = np.array(clip)
                                    clip_tensor = torch.from_numpy(np_array)
                                    clips_tensors.append(clip_tensor)
                                except Exception as np_error:
                                    # print(f"[ERROR] Failed numpy conversion for clip {j}: {np_error}")
                                    raise ValueError(f"Cannot convert clip {j} to tensor")
                        else:
                            # print(f"[ERROR] Unexpected clip type: {type(clip)}")
                            raise ValueError(f"Invalid clip type: {type(clip)}")
                    
                    # Stack clips along a new dimension: [num_clips, C, T, H, W]
                    if clips_tensors:
                        stacked_clips = torch.stack(clips_tensors, dim=0)
                        processed_video.append(stacked_clips)
                    else:
                        raise ValueError(f"No valid clips found for sample {i}")
                else:
                    # print(f"[ERROR] Invalid clips_list for sample {i}: {type(clips_list)}")
                    raise ValueError(f"Invalid clips_list type: {type(clips_list)}")
            except Exception as e:
                # print(f"[ERROR] Failed processing video clips {i}: {e}")
                raise

        # Stack all samples: [B, num_clips, C, T, H, W]
        video = torch.stack(processed_video)  # [B, num_clips, C, T, H, W]
        
        # For now, let's use the first clip only: [B, C, T, H, W]
        video = video[:, 0]  # Take first clip from each sample

        # Move video to the same device as the model
        # Get device from the global encoder that patch_embed_fn uses
        try:
            # Import the global encoder from the training module
            import app.vjepa.train as train_module
            device = next(train_module._global_patch_embed_encoder.parameters()).device
        except:
            # Fallback: use CUDA if available, otherwise CPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        video = video.to(device)

        with torch.no_grad():
            x = self.patch_embed_fn(video)  # [B, N, D]

        # Dynamically initialize components based on the actual model dimension
        if not self.initialized:
            self._initialize_components(x.size(-1), x.device)  # Pass both model_dim and device

        x = x + self.pos_embed_probs.to(x.device)

        logits = self.get_token_probs(x)
        logits = torch.nan_to_num(logits)
        p_x = self.softmax(logits)  # [B, N]
        # Clamp and normalize probabilities to avoid invalid values
        p_x = torch.clamp(p_x, min=1e-8)
        p_x = p_x / p_x.sum(dim=1, keepdim=True)
        self.last_mask_probs = p_x.detach()
        
        # Store importance scores for visualization
        self.last_importance_scores = p_x.detach()

        # Return masks in the same format as the reference MaskCollator
        # mask_e: indices of masked tokens, mask_p: indices of visible tokens
        masks_enc = []
        masks_pred = []
        
        for i in range(batch_size):
            # Get visible indices for this sample
            vis_idx = torch.multinomial(p_x[i], num_samples=self.visible_patches, replacement=False)

            # Create mask indices (similar to reference implementation)
            all_indices = torch.arange(self.num_tokens, device=x.device)
            mask_e = all_indices[~torch.isin(all_indices, vis_idx)]  # masked indices
            mask_p = vis_idx  # visible indices
            
            # Ensure 1D tensors even if only one element or empty
            if mask_e.numel() == 0:
                # If no masked indices, create an empty 1D tensor
                mask_e = torch.empty(0, dtype=torch.long, device=x.device)
            elif mask_e.dim() == 0:
                mask_e = mask_e.unsqueeze(0)
            else:
                mask_e = mask_e.view(-1)  # Ensure 1D
            
            if mask_p.numel() == 0:
                # If no visible indices, create an empty 1D tensor
                mask_p = torch.empty(0, dtype=torch.long, device=x.device)
            elif mask_p.dim() == 0:
                mask_p = mask_p.unsqueeze(0)
            else:
                mask_p = mask_p.view(-1)  # Ensure 1D
                
            # print(f"[DEBUG] Sample {i}: mask_e shape: {mask_e.shape}, mask_p shape: {mask_p.shape}")

            # Move to CPU for proper pin_memory handling
            masks_enc.append(mask_e.cpu())
            masks_pred.append(mask_p.cpu())
            
            # Debug: print tensor shapes
            # print(f"[DEBUG] Sample {i}: mask_e shape: {mask_e.shape}, mask_p shape: {mask_p.shape}")
            # print(f"[DEBUG] Sample {i}: mask_e dims: {mask_e.dim()}, mask_p dims: {mask_p.dim()}")
            # print(f"[DEBUG] Sample {i}: mask_e numel: {mask_e.numel()}, mask_p numel: {mask_p.numel()}")
        
        # Collate the masks
        masks_enc = torch.utils.data.default_collate(masks_enc)
        masks_pred = torch.utils.data.default_collate(masks_pred)
        
        # Store masks for visualization
        self.last_masks = (masks_enc, masks_pred)
        
        # Store the number of visible patches for REINFORCE sampling in train.py
        self.last_visible_patches = self.visible_patches
        
        # Suppose you have p_x: [B, N] (probabilities for each token)
        # and mask_p: [B, N_visible] (indices of visible tokens for each sample)

        # Compute log-probs for all tokens
        log_probs = torch.log(p_x + 1e-8)  # [B, N]

        # For each sample, get log-probs for the selected (visible) tokens
        mask_log_probs = []
        for i in range(batch_size):
            # p_x: [B, N] (probabilities for each token)
            sampled_indices = torch.multinomial(p_x[i], num_samples=self.visible_patches, replacement=False)
            log_probs = torch.log(p_x[i] + 1e-8)
            mask_log_probs.append(log_probs[sampled_indices])  # [N_visible]
        mask_log_probs = torch.stack(mask_log_probs)  # [B, N_visible]
        self.last_mask_log_probs = mask_log_probs

        # You will also need to compute and store:
        # - self.last_mask_l_r: the per-token reconstruction error (from your model, after forward)
        # - self.last_bool_masked_pos: a boolean mask or indices for masked positions (if needed)
        
        # Properly collate the batch data into tensors
        collated_batch = torch.utils.data.default_collate(batch)
        
        # print(f"[DEBUG] masks_enc: {len(masks_enc)}, masks_pred: {len(masks_pred)}")
        
        # For adaptive masking, return multiple dynamic masks with different strategies
        # Strategy 1: High information regions (like AdaMAE)
        masks_enc_list = [masks_enc]
        masks_pred_list = [masks_pred]
        
        # Strategy 2: Different sampling ratio for temporal focus
        # Create a different mask with different sampling ratio
        if len(masks_enc) > 0 and len(masks_pred) > 0:
            # Create a second mask with different sampling strategy
            # For now, we'll use the same mask but in practice this could be different
            # This avoids memory duplication while maintaining the multi-mask structure
            masks_enc_list.append(masks_enc)
            masks_pred_list.append(masks_pred)
        
        # Visualize if enabled
        if self.visualize and self.last_importance_scores is not None:
            self.visualize_masking(video, self.last_importance_scores, 
                                 masks_enc_list, masks_pred_list)
        
        return collated_batch, masks_enc_list, masks_pred_list
# ..............
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import default_collate


# class AdaptiveMaskCollator(nn.Module):
#     """
#     AdaMAE-style masking with dynamic number of visible tokens per sample.
#     Retains enough tokens to reach a cumulative importance threshold (e.g., 25%).
#     """
#     def __init__(self, num_tokens, patch_embed_fn, threshold=0.25, visualize=False):
#         super().__init__()
#         self.num_tokens = num_tokens
#         self.threshold = threshold
#         self.patch_embed_fn = patch_embed_fn
#         self.visualize = visualize

#         self.token_attn = None
#         self.linear = None
#         self.softmax = nn.Softmax(dim=-1)
#         self.initialized = False

#         self.last_importance_scores = None
#         self.last_masks = None
#         self.last_visible_counts = None

#     def _initialize_components(self, dim, device):
#         self.token_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True).to(device)
#         self.linear = nn.Linear(dim, self.num_tokens).to(device)
#         self.initialized = True

#     def get_token_probs(self, x):
#         z, _ = self.token_attn(x, x, x)
#         logits = self.linear(z.mean(dim=1))  # [B, N]
#         return self.softmax(logits)

#     def step(self):
#         pass  # for JEPA compatibility

#     def __call__(self, batch):
#         batch_size = len(batch)

#         # Extract video tensor [B, C, T, H, W]
#         videos = torch.stack([x[0][0] if isinstance(x[0], list) else x[0] for x in batch])
#         device = videos.device

#         with torch.no_grad():
#             x = self.patch_embed_fn(videos)  # [B, N, D]


#         if not self.initialized:
#             self._initialize_components(x.size(-1), x.device)

#         probs = self.get_token_probs(x)  # [B, N]
#         self.last_importance_scores = probs.detach().clone()

#         masks_enc, masks_pred, visible_counts = [], [], []

#         for i in range(batch_size):
#             prob_i = probs[i]  # [N]
#             sorted_probs, sorted_idx = torch.sort(prob_i, descending=True)
#             cumsum = torch.cumsum(sorted_probs, dim=0)
#             k = (cumsum < self.threshold).sum().item() + 1
#             visible_idx = sorted_idx[:k]
#             mask_idx = torch.arange(self.num_tokens, device=device)
#             visible_idx = visible_idx.to(mask_idx.device)
#             mask_idx = mask_idx[~torch.isin(mask_idx, visible_idx)]

#             masks_enc.append(mask_idx.cpu())
#             masks_pred.append(visible_idx.cpu())
#             visible_counts.append(k)

#         # Save last masks for visualization
#         self.last_masks = (masks_enc, masks_pred)
#         self.last_visible_counts = visible_counts

#         # JEPA expects a list of masks (possibly more than 1 strategy)
#         masks_enc = default_collate(masks_enc)
#         masks_pred = default_collate(masks_pred)

#         # return default_collate(batch), [masks_enc], [masks_pred]
#         return batch, masks_enc, masks_pred  # as list of tensors

