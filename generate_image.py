import torch
import os
from diffusers import AutoPipelineForText2Image

# PEFT 라이브러리 필요 (LoRA 로딩용)
from peft import PeftModel, PeftConfig

from huggingface_hub import login

login(os.environ.get("HUGGING_API_KEY"))

# 기기 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# 기본 모델 로드
print("기본 FLUX 모델 로드 중...")
pipe = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.float16  # bfloat16 대신 float16 사용
)
pipe.enable_model_cpu_offload()
pipe.to(device)

# Uncensored LoRA 로드
print("Uncensored LoRA 로드 중...")
pipe.load_lora_weights(
    'Heartsync/Flux-NSFW-uncensored', 
    weight_name='lora.safetensors',
    adapter_name="uncensored"
)

# 이미지 생성
prompt = sys.argv[1]
negative_prompt = "text, watermark, signature, cartoon, anime, illustration, painting, drawing, low quality, blurry"

# 시드 설정
seed = 42
generator = torch.Generator(device=device).manual_seed(seed)

# 이미지 생성
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=7.0,
    num_inference_steps=28,
    width=544,
    height=960,
    generator=generator,
).images[0]

# 이미지 저장
image.save("generated_image.png")
