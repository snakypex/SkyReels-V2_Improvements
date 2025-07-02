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
import requests

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

API_URL = "https://liroai.com/api/getpendinggeneration"
DEFAULT_PROMPT = (
    "Cute cartoon doctor with shiny blue hair, gold glasses and stethoscope, performing an allergy skin test on a patient’s arm. Bright, vibrant colors, glossy 3D plastic toy style. Close-up shot of the doctor gently pricking the patient’s forearm with a pen-like tool, small droplets or dots appearing on the skin. The background shows a colorful cross-section of skin layers with hair follicles and nerves. Cheerful, educational atmosphere. Camera slowly zooms in and pans from doctor’s face to the patient’s arm. Smooth, fluid animation, bright lighting, no text."
)


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

    parser.add_argument("--batch_size", type=int, default=1) # 20250422 pftq: Batch functionality to avoid reloading the model each video
    parser.add_argument("--preserve_image_aspect_ratio", action="store_true")  # 20250422 pftq: Avoid resizing
    parser.add_argument("--negative_prompt", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards") # 20250422 pftq: expose negative prompt
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
    )
    parser.add_argument("--prompt_enhancer", action="store_true")
    parser.add_argument("--teacache", action="store_true")
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="Higher speedup will cause to worse quality -- 0.1 for 2.0x speedup -- 0.2 for 3.0x speedup")
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        help="Using Retention Steps will result in faster generation speed and better generation quality.")
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

    # juste après la section resolution
    orig_width, orig_height = width, height

    #image = load_image(args.image).convert("RGB") if args.image else None

        
    #20250422 pftq: Add error handling for image loading, aspect ratio preservation
    image = None
    if args.image:  
        try:
            image = load_image(args.image).convert("RGB")

            # 20250422 pftq: option to preserve image aspect ratio
            if args.preserve_image_aspect_ratio:
                img_width, img_height = image.size
                if img_height > img_width:
                    height, width = width, height
                    width = int(height / img_height * img_width)
                else:
                    height = int(width / img_width * img_height)

                divisibility=16
                if width%divisibility!=0:
                        width = width - (width%divisibility)
                if height%divisibility!=0:
                        height = height - (height%divisibility)

                image = resizecrop(image, height, width)
            else:
                image_width, image_height = image.size
                if image_height > image_width:
                    height, width = width, height
                image = resizecrop(image, height, width)
        except Exception as e:
            raise ValueError(f"Failed to load or process image: {e}")

    print(f"Rank {local_rank}: {width}x{height} | Image: "+str(image!=None))
    
    negative_prompt = args.negative_prompt  # 20250422 pftq: allow editable negative prompt

    def enhance_prompt(text: str) -> str:
        if args.prompt_enhancer:
            print("init prompt enhancer")
            prompt_enhancer = PromptEnhancer()
            text = prompt_enhancer(text)
            print(f"enhanced prompt: {text}")
            del prompt_enhancer
            gc.collect()
            torch.cuda.empty_cache()
        return text

    prompt_input = enhance_prompt(args.prompt)

    # 20250423 pftq: needs fixing, 20-min load times on multi-GPU caused by contention, DF already reduced down to 12 min roughly the same as single GPU.
    print("Initializing pipe at "+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    starttime = time.time()
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
    totaltime = time.time()-starttime
    print("Finished initializing pipe at "+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())+" ("+str(int(totaltime))+" seconds)")


    if args.teacache:
        pipe.transformer.initialize_teacache(
            enable_teacache=True,
            num_steps=args.inference_steps,
            teacache_thresh=args.teacache_thresh,
            use_ret_steps=args.use_ret_steps,
            ckpt_dir=args.model_id,
        )

    torch.backends.cuda.preferred_linalg_library("default")  # or try "magma" if available

    def generate_once(img, p_text):
        for idx in range(args.batch_size):
            if local_rank == 0:
                print(f"Generating video {idx+1} of {args.batch_size}")

            if args.use_usp:
                try:
                    dist.barrier()
                    if local_rank == 0:
                        if args.seed == -1 or idx > 0:
                            args.seed = int(random.randrange(4294967294))
                    seed_tensor = torch.tensor(args.seed, dtype=torch.int64, device="cuda")
                    dist.broadcast(seed_tensor, src=0)
                    args.seed = seed_tensor.item()
                    dist.barrier()
                except Exception as e:
                    print(f"[Rank {local_rank}] Seed broadcasting error: {e}")
                    dist.destroy_process_group()
                    raise
            else:
                if args.seed == -1 or idx > 0:
                    args.seed = int(random.randrange(4294967294))

            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

            kwargs = {
                "prompt": p_text,
                "negative_prompt": negative_prompt,
                "num_frames": args.num_frames,
                "num_inference_steps": args.inference_steps,
                "guidance_scale": args.guidance_scale,
                "shift": args.shift,
                "generator": torch.Generator(device="cuda").manual_seed(args.seed),
                "height": orig_height,
                "width": orig_width
            }

            if img is not None:
                kwargs["image"] = img

            save_dir = os.path.join("result", args.outdir)
            os.makedirs(save_dir, exist_ok=True)

            with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
                print(f"infer kwargs:{kwargs}")
                video_frames = pipe(**kwargs)[0]

            if local_rank == 0:
                current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                gpucount = ""
                if args.use_usp and dist.get_world_size():
                    gpucount = "_" + str(dist.get_world_size()) + "xGPU"
                video_out_file = (
                    f"{current_time}_skyreels2_{args.resolution}-{args.num_frames}f_cfg{args.guidance_scale}_steps{args.inference_steps}_seed{args.seed}{gpucount}_{p_text[:100].replace('/', '')}_{idx}.mp4"
                )
                output_path = os.path.join(save_dir, video_out_file)
                imageio.mimwrite(
                    output_path,
                    video_frames,
                    fps=args.fps,
                    quality=8,
                    output_params=["-loglevel", "error"],
                )

    def fetch_task():
        try:
            resp = requests.get(API_URL, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "image_url" in data and "prompt" in data:
                    return data["image_url"], data["prompt"]
        except Exception as e:
            print(f"API request failed: {e}")
        return None, None

    while True:
        task_image_url, task_prompt = fetch_task()
        if not task_image_url or not task_prompt:
            time.sleep(1)
            continue

        try:
            img = load_image(task_image_url).convert("RGB")
    
            # pour chaque itération, on repart des valeurs d'origine
            w, h = orig_width, orig_height
    
            if args.preserve_image_aspect_ratio:
                img_w, img_h = img.size
                if img_h > img_w:
                    # swap local
                    h, w = w, h
                # recalc local en gardant aspect
                w = int(h / img_h * img_w)
                # assurons la divisibilité 16
                w -= w % 16
                h -= h % 16
                img = resizecrop(img, h, w)
            else:
                img_w, img_h = img.size
                if img_h > img_w:
                    # swap local
                    h, w = w, h
                img = resizecrop(img, h, w)
        except Exception as e:
            print(f"Failed to load or process image: {e}")
            time.sleep(1)
            continue

        generate_once(img, task_prompt)
