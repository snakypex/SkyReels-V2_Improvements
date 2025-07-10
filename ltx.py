import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video

pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.7-dev", torch_dtype=torch.bfloat16)
pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained("Lightricks/ltxv-spatial-upscaler-0.9.7", vae=pipe.vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe_upsample.to("cuda")
pipe.vae.enable_tiling()

def round_to_nearest_resolution_acceptable_by_vae(height, width):
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width

image = load_image("https://i.postimg.cc/TwQV5Ppg/generated-image-2.png")
video = load_video(export_to_video([image])) # compress the image using video compression as the model was trained on videos
condition1 = LTXVideoCondition(video=video, frame_index=0)

prompt = "FPS-24, A young woman with long blonde hair sits on a bed, her legs spread apart as she faces the camera with a subtle, inviting smile. Her arms rest gently on her thighs, and her gaze is direct, engaging the viewer. The plain white wall behind her provides a clean, minimalist backdrop, allowing her relaxed, natural pose to take center stage. Her hair cascades down her back, framing her calm, serene expression as she embodies comfort and confidence."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
expected_height, expected_width = 960, 544
downscale_factor = 2 / 3
num_frames = 96

# Part 1. Generate video at smaller resolution
downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
latents = pipe(
    conditions=[condition1],
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=downscaled_width,
    height=downscaled_height,
    num_frames=num_frames,
    num_inference_steps=30,
    generator=torch.Generator().manual_seed(0),
    output_type="latent",
).frames

# Part 2. Upscale generated video using latent upsampler with fewer inference steps
# The available latent upsampler upscales the height/width by 2x
upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
upscaled_latents = pipe_upsample(
    latents=latents,
    output_type="latent"
).frames

# Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
video = pipe(
    conditions=[condition1],
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=upscaled_width,
    height=upscaled_height,
    num_frames=num_frames,
    denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
    num_inference_steps=10,
    latents=upscaled_latents,
    decode_timestep=0.05,
    image_cond_noise_scale=0.025,
    generator=torch.Generator().manual_seed(0),
    output_type="pil",
).frames[0]

# Part 4. Downscale the video to the expected resolution
video = [frame.resize((expected_width, expected_height)) for frame in video]

export_to_video(video, "output.mp4", fps=24)
