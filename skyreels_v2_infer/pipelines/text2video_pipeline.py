import os
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm

from ..modules import get_text_encoder
from ..modules import get_transformer
from ..modules import get_vae
from ..scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler


class Text2VideoPipeline:
    def __init__(
        self, model_path, dit_path, device: str = "cuda", weight_dtype=torch.bfloat16, use_usp=False, offload=False
    ):
        # 20250423 pftq: Fixed load time by broadcasting transformer and staggering text encoder, VAE
        import torch.distributed as dist
        load_device = "cpu" if offload else device
        self.device = device
        self.offload = offload

        # 20250423 pftq: Check rank and distributed mode
        if use_usp:
            if not dist.is_initialized():
                raise RuntimeError("Distributed environment must be initialized with dist.init_process_group before using use_usp=True")
            local_rank = dist.get_rank()
        else:
            local_rank = 0

        print(f"[Rank {local_rank}] Initializing pipeline components...")

        # 20250423 pftq: Load transformer only on rank 0 or single-GPU
        if not use_usp or local_rank == 0:
            print(f"[Rank {local_rank}] Loading transformer...")
            self.transformer = get_transformer(dit_path, load_device, weight_dtype, skip_weights=False)
            transformer_state_dict = self.transformer.state_dict() if use_usp else None
        else:
            print(f"[Rank {local_rank}] Skipping weights for transformer...")
            self.transformer = get_transformer(dit_path, load_device, weight_dtype, skip_weights=True)
            transformer_state_dict = None

        # 20250423 pftq: Load text encoder only on rank 0 or single-GPU
        if not use_usp or local_rank == 0:
            print(f"[Rank {local_rank}] Loading text encoder...")
            self.text_encoder = get_text_encoder(model_path, load_device, weight_dtype, skip_weights=False)
            text_encoder_state_dict = self.text_encoder.state_dict() if use_usp else None
        else:
            print(f"[Rank {local_rank}] Initializing empty text encoder...")
            self.text_encoder = get_text_encoder(model_path, load_device, weight_dtype, skip_weights=True)
            text_encoder_state_dict = None

        # 20250423 pftq: Broadcast transformer and text encoder weights from rank 0
        if use_usp:
            dist.barrier()  # Ensure rank 0 loads transformer and text encoder
            broadcast_list = [transformer_state_dict, text_encoder_state_dict]
            print(f"[Rank {local_rank}] Broadcasting weights for transformer and text encoder...")
            dist.broadcast_object_list(broadcast_list, src=0)
            transformer_state_dict, text_encoder_state_dict = broadcast_list
            print(f"[Rank {local_rank}] Loading broadcasted transformer and text encoder weights...")
            self.transformer.load_state_dict(transformer_state_dict)
            self.text_encoder.load_state_dict(text_encoder_state_dict)
            dist.barrier()  # Synchronize ranks

        # 20250423 pftq: Stagger VAE loading across ranks
        vae_model_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        if use_usp:
            for rank in range(dist.get_world_size()):
                if local_rank == rank:
                    print(f"[Rank {local_rank}] Loading VAE...")
                    self.vae = get_vae(vae_model_path, device, weight_dtype=torch.float32)
                dist.barrier()
        else:
            print(f"[Rank {local_rank}] Loading VAE...")
            self.vae = get_vae(vae_model_path, device, weight_dtype=torch.float32)

        self.video_processor = VideoProcessor(vae_scale_factor=16)
        self.sp_size = 1
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward
            import types

            for block in self.transformer.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
                # 20250423 pftq: Fixed indentation and removed duplicate forward assignment
                self.transformer.forward = types.MethodType(usp_dit_forward, self.transformer)
            self.sp_size = get_sequence_parallel_world_size()

        self.scheduler = FlowUniPCMultistepScheduler()
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        width: int = 544,
        height: int = 960,
        num_frames: int = 97,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        shift: float = 5.0,
        generator: Optional[torch.Generator] = None,
    ):
        # preprocess
        F = num_frames
        target_shape = (
            self.vae.vae.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            height // self.vae_stride[1],
            width // self.vae_stride[2],
        )
        self.text_encoder.to(self.device)
        context = self.text_encoder.encode(prompt).to(self.device)
        context_null = self.text_encoder.encode(negative_prompt).to(self.device)
        if self.offload:
            self.text_encoder.cpu()
            torch.cuda.empty_cache()

        latents = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            )
        ]

        # evaluation mode
        self.transformer.to(self.device)
        with torch.cuda.amp.autocast(dtype=self.transformer.dtype), torch.no_grad():
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, shift=shift)
            timesteps = self.scheduler.timesteps

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = torch.stack(latents)
                timestep = torch.stack([t])
                noise_pred_cond = self.transformer(latent_model_input, t=timestep, context=context)[0]
                noise_pred_uncond = self.transformer(latent_model_input, t=timestep, context=context_null)[0]

                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = self.scheduler.step(
                    noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=generator
                )[0]
                latents = [temp_x0.squeeze(0)]
            if self.offload:
                self.transformer.cpu()
                torch.cuda.empty_cache()
            videos = self.vae.decode(latents[0])
            videos = (videos / 2 + 0.5).clamp(0, 1)
            videos = [video for video in videos]
            videos = [video.permute(1, 2, 3, 0) * 255 for video in videos]
            videos = [video.cpu().numpy().astype(np.uint8) for video in videos]
        return videos
