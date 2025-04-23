import argparse
import gc
import os
import random
import time

import imageio
import torch
from diffusers.utils import load_image

from PIL import Image  #20250422 pftq: Added for image resizing and cropping
import numpy as np  #20250422 pftq: Added for seed synchronization

from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Image2VideoPipeline
from skyreels_v2_infer.pipelines import PromptEnhancer
from skyreels_v2_infer.pipelines import resizecrop
from skyreels_v2_infer.pipelines import Text2VideoPipeline

MODEL_ID_CONFIG = {
    "text2video": [
        "Skywork/SkyReels-V2-T2V-14B-540P",
        "Skywork/SkyReels-V2-T2V-14B-720P",
    ],
    "image2video": [
        "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "Skywork/SkyReels-V2-I2V-14B-540P",
        "Skywork/SkyReels-V2-I2V-14B-720P",
    ],
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="video_out")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-T2V-14B-540P")
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"])
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--use_usp", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface.",
    )
    parser.add_argument("--prompt_enhancer", action="store_true")

    parser.add_argument("--batch_size", type=int, default=1) # 20250422 pftq: Batch functionality to avoid reloading the model each video
    parser.add_argument("--preserve_image_aspect_ratio", action="store_true")  # 20250422 pftq: Avoid resizing
    parser.add_argument("--negative_prompt", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards") # 20250422 pftq: expose negative prompt
    
    args = parser.parse_args()

    args.model_id = download_model(args.model_id)
    print("model_id:", args.model_id)

    #20250422 pftq: unneeded with seed synchronization code
    #assert (args.use_usp and args.seed is not None) or (not args.use_usp), "usp mode need seed"

    local_rank = 0
    if args.use_usp:
        assert not args.prompt_enhancer, "`--prompt_enhancer` is not allowed if using `--use_usp`. We recommend running the skyreels_v2_infer/pipelines/prompt_enhancer.py script first to generate enhanced prompt before enabling the `--use_usp` parameter."
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        import torch.distributed as dist

        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(dist.get_rank())
        device = "cuda"

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )

    if args.resolution == "540P":
        height = 544
        width = 960
    elif args.resolution == "720P":
        height = 720
        width = 1280
    else:
        raise ValueError(f"Invalid resolution: {args.resolution}")

    #image = load_image(args.image).convert("RGB") if args.image else None

    #20250422 pftq: Add error handling for image loading, aspect ratio preservation, and multi-GPU synchronization
    image = None
    if args.image:
        if local_rank == 0:
            try:
                image = load_image(args.image).convert("RGB")

                # 20250422 pftq: option to preserve image aspect ratio
                if args.preserve_image_aspect_ratio:
                    img_width, img_height = image.size
                    height = int(width / img_width * img_height)
                else:
                    image_width, image_height = image.size
                    if image_height > image_width:
                        height, width = width, height
                    image = resizecrop(image, height, width)
            except Exception as e:
                raise ValueError(f"Failed to load or process image: {e}")
                
        if args.use_usp:
            # Broadcast image to other ranks
            image_data = torch.tensor(np.array(image), dtype=torch.uint8, device="cuda") if image is not None else None
            if local_rank == 0:
                dist.broadcast(image_data, src=0)
            else:
                image_data = torch.empty((height, width, 3), dtype=torch.uint8, device="cuda")
                dist.broadcast(image_data, src=0)
                image = Image.fromarray(image_data.cpu().numpy())

            #20250422 pftq: Broadcast height and width to ensure consistency
            height_tensor = torch.tensor(height, dtype=torch.int64, device="cuda")
            width_tensor = torch.tensor(width, dtype=torch.int64, device="cuda")
            dist.broadcast(height_tensor, src=0)
            dist.broadcast(width_tensor, src=0)
            height = height_tensor.item()
            width = width_tensor.item()

    negative_prompt = args.negative_prompt # 20250422 pftq: allow editable negative prompt

    prompt_input = args.prompt
    if args.prompt_enhancer and args.image is None:
        print(f"init prompt enhancer")
        prompt_enhancer = PromptEnhancer()
        prompt_input = prompt_enhancer(prompt_input)
        print(f"enhanced prompt: {prompt_input}")
        del prompt_enhancer
        gc.collect()
        torch.cuda.empty_cache()

    if image is None:
        assert "T2V" in args.model_id, f"check model_id:{args.model_id}"
        print("init text2video pipeline")
        pipe = Text2VideoPipeline(
            model_path=args.model_id, dit_path=args.model_id, use_usp=args.use_usp, offload=args.offload
        )
    else:
        assert "I2V" in args.model_id, f"check model_id:{args.model_id}"
        print("init img2video pipeline")
        pipe = Image2VideoPipeline(
            model_path=args.model_id, dit_path=args.model_id, use_usp=args.use_usp, offload=args.offload
        )

    prompt_input = args.prompt
    if args.prompt_enhancer and image is not None:
        prompt_input = prompt_enhancer(prompt_input)
        print(f"enhanced prompt: {prompt_input}")


    #20250422 pftq: Set preferred linear algebra backend to avoid cuSOLVER issues
    torch.backends.cuda.preferred_linalg_library("default")  # or try "magma" if available
    
    for idx in range(args.batch_size): # 20250422 pftq: implemented --batch_size
        if local_rank == 0:
            print(f"Generating video {idx+1} of {args.batch_size}")

        #20250422 pftq: Synchronize seed across all ranks
        if args.use_usp:
            try:
                #20250422 pftq: Synchronize ranks before seed broadcasting
                dist.barrier()

                #20250422 pftq: Always broadcast seed to ensure consistency
                if local_rank == 0:
                    if args.seed == -1 or args.seed is None or idx > 0:
                        args.seed = int(random.randrange(4294967294))
                seed_tensor = torch.tensor(args.seed, dtype=torch.int64, device="cuda")
                dist.broadcast(seed_tensor, src=0)
                args.seed = seed_tensor.item()

                #20250422 pftq: Synchronize ranks after seed broadcasting
                dist.barrier()
            except Exception as e:
                print(f"[Rank {local_rank}] Seed broadcasting error: {e}")
                dist.destroy_process_group()
                raise

        else:
            #20250422 pftq: Single GPU seed initialization
            if args.seed == -1 or idx > 0:
                args.seed = int(random.randrange(4294967294))

        #20250422 pftq: Set seeds for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        kwargs = {
            "prompt": prompt_input,
            "negative_prompt": negative_prompt,
            "num_frames": args.num_frames,
            "num_inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "shift": args.shift,
            "generator": torch.Generator(device="cuda").manual_seed(args.seed),
            "height": height,
            "width": width,
        }
    
        if image is not None:
            #kwargs["image"] = load_image(args.image).convert("RGB") 
            # 20250422 pftq: redundant reloading of the image
            kwargs["image"] = image
    
        save_dir = os.path.join("result", args.outdir)
        os.makedirs(save_dir, exist_ok=True)
    
        with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
            print(f"infer kwargs:{kwargs}")
            video_frames = pipe(**kwargs)[0]
    
        if local_rank == 0:
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            #video_out_file = f"{args.prompt[:100].replace('/','')}_{args.seed}_{current_time}.mp4"
            
            # 20250422 pftq: more useful filename
            gpucount = ""
            if args.use_usp and dist.get_world_size():
                gpucount = "_"+str(dist.get_world_size())+"xGPU"
            video_out_file = f"{current_time}_skyreels2_cfg{args.guidance_scale}_steps{args.inference_steps}_seed{args.seed}{gpucount}_{args.prompt[:100].replace('/','')}_{idx}.mp4" 
            
            output_path = os.path.join(save_dir, video_out_file)
            imageio.mimwrite(output_path, video_frames, fps=args.fps, quality=8, output_params=["-loglevel", "error"])
