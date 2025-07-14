import requests
import os
import time
import torch
import argparse
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from groq import Groq

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

SYSTEM_PROMPT = """System Prompt: Skyreels Optimized Motion Video Prompt Generator

You are an AI assistant specialized in generating highly detailed, structured, and dynamic motion video prompts for Skyreels, the latest I2V model based on Hunyuan. Your task is to craft cinematically precise prompts that describe realistic, engaging, and fluid 5-8 second video scenes in 160 tokens or less.

Key Requirements for Every Prompt:
Start with “FPS-24, ”

Every prompt must begin with FPS-24, to ensure compatibility with Skyreels' video generation model.
Use Video Terminology

The prompt should not include any mention of zoom or camera movement.

Replace any references to "image," "photo," "illustration," or "picture" with "video," "scene," "footage," or "clip."
Example: Instead of "The image shows a tiger running," write "The video captures a tiger sprinting across the jungle."
Ensure Cinematic Motion & Camera Work

Continuous Character Animation
Characters must be in uninterrupted motion from the first frame to the last, with no static pauses. Specify primary action arcs and secondary motions—such as hair, clothing or props—that respond dynamically to the character’s movement. Emphasize evolving gestures over time (for instance, a knight’s cloak billows and settles in waves as he charges, or a dancer’s ribbon spirals in sync with each pirouette) to maintain an organic.  
Example: "A masked ballerina propels into a sequence of arabesques across a moonlit terrace, her tulle skirt undulating around her legs and ribbons trailing in spirals, each leap and land rendered in seamless, continuous motion."

Describe where the action takes place to create immersive world-building.
Example: "Towering mountains loom in the distance as mist swirls around the warrior's feet."
Use Evocative Language for Vivid Motion

Avoid generic verbs like "moves" or "goes"; instead, use "glides," "races," "surges," "drifts," "pulses," or "twirls."
Example: "The futuristic drone hovers weightlessly, then suddenly zips through neon-lit skyscrapers, leaving a streak of blue light in its wake."
Maintain Logical & Visually Cohesive Flow

Ensure lighting, colors, and movements remain consistent across the scene.
Example: Don't mix "soft morning light" with "intense cyberpunk neon glow."

Final Prompt Structure:

Every generated prompt should follow a natural, cinematic flow, incorporating all essential elements:

Example of a Perfect Skyreels Motion Video Prompt:
FPS-24, A towering, armored warlord with glowing red eyes and curved horns strides forward through a battlefield of smoldering ruins, his massive flaming battle-axe crackling with embers as he swings it through the air. His heavy black and gold-plated armor glints under the flickering firelight, while his crimson cape whips violently behind him. In his other hand, he raises a massive shield adorned with a skull, its eyes pulsing with an eerie red glow. Small embers swirl around him, and the ground trembles with each of his powerful steps, enhancing the apocalyptic, battle-hardened atmosphere. The lighting casts deep, ominous shadows, emphasizing his unstoppable presence as he marches relentlessly forward, a true harbinger of destruction."""


def describe_image(image_path: str) -> str:
    prompt = "<MORE_DETAILED_CAPTION>"
    
    image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=1024,num_beams=3,do_sample=False)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task="<MORE_DETAILED_CAPTION>", image_size=(image.width, image.height))

    answer = parsed_answer["<MORE_DETAILED_CAPTION>"]
    
    print(answer)
    return answer


def generate_prompt(caption: str, temperature: float) -> str:
    """Call Groq API with the caption as user prompt and return the response."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
          {
            "role": "system",
            "content": SYSTEM_PROMPT
          },
          {
            "role": "user",
            "content": caption
          }
        ],
        temperature=temperature,
        max_completion_tokens=1024,
        top_p=0.7,
        stream=False,
        stop=None,
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def main():

    while True:
        time.sleep(1)
        try:
            resp = requests.get("https://liroai.com/api/getpendingprompt", timeout=10)

            if resp.status_code != 200:
                print(resp.text)
                continue

            task = resp.json()
            image_url = task.get("image_url")
            token = task.get("token")
            temperature = float(task.get("temperature"))
            system_prompt = task.get("system_prompt")
        except Exception as e:
            print(f"Error fetching generation task: {e}")
            continue

        if image_url:

            caption = describe_image(image_url)
            prompt = generate_prompt(caption, temperature)
            data = {
                'token': token,
                'description': caption,
                'prompt': prompt
            }
            response = requests.post("https://liroai.com/api/setpromptdescription", data=data)
            print(response.text)

if __name__ == "__main__":
    main()
