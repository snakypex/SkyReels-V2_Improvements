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


def generate_prompt(caption: str, temperature: float, system_prompt: str) -> str:
    """Call Groq API with the caption as user prompt and return the response."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
          {
            "role": "system",
            "content": system_prompt
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
            prompt = generate_prompt(caption, temperature, system_prompt)
            data = {
                'token': token,
                'description': caption,
                'prompt': prompt
            }
            response = requests.post("https://liroai.com/api/setpromptdescription", data=data)
            print(response.text)

if __name__ == "__main__":
    main()
