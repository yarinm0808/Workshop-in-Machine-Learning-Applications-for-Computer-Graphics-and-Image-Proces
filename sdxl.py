"""
Dynamic Stage-Aware Prompt Injection Pipeline (Single Image Mode)

Architecture:
- GPU 0: Handles the Stable Diffusion XL (SDXL) generation process.
- GPU 1: Handles the Vision-Language Model (Qwen2-VL) used for real-time visual feedback.
"""

import os
# Force Hugging Face to download models to a local folder you own to prevent quota issues
os.environ["HF_HOME"] = "./.hidden_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import torch
import random
import json
import gc
import time
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw, ImageFont

# Create necessary output directories
os.makedirs("debug_steps", exist_ok=True)
os.makedirs("results", exist_ok=True) # Final outputs and logs

def print_gpu_status(label=""):
    """Utility function to monitor and print current GPU VRAM allocation."""
    print(f"\n[GPU STATUS] {label}")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {i}: Used: {allocated:.2f} GB")
    print("-" * 30)

# ==========================================
# 1. VLM Setup & Queries (Runs on GPU 1)
# ==========================================

def load_vlm():
    print("Loading Qwen2-VL-2B-Instruct to GPU 1...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", 
        torch_dtype=torch.float16, 
        device_map="cuda:1"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return processor, model

def query_vlm_dual_gpu(processor, model, image, question=None):
    device = model.device 
    if question is None:
        question = "Describe the image."

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
    return output_text

# ==========================================
# 2. Diffusion Helpers (Runs on GPU 0)
# ==========================================

def decode_image(pipe, tensor_to_decode):
    if pipe.vae.device.type != "cuda":
        pipe.vae.to("cuda:0")
    tensor_to_decode = tensor_to_decode.to(pipe.vae.dtype).to("cuda:0")
    with torch.no_grad():
        decoded = pipe.vae.decode(tensor_to_decode / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    return image

def create_phantom_comparison(pipe, latents, t, proxy_current, proxy_next, add_time_ids, step_num):
    device = latents.device
    
    def get_projection(proxy):
        noise_pred = pipe.unet(
            torch.cat([latents] * 2).to(device), t, 
            encoder_hidden_states=torch.cat([proxy["neg"], proxy["pos"]], dim=0),
            added_cond_kwargs={"text_embeds": torch.cat([proxy["pooled_neg"], proxy["pooled_pos"]], dim=0), "time_ids": add_time_ids},
            return_dict=False
        )[0]
        u, c = noise_pred.chunk(2)
        noise_pred = u + 7.5 * (c - u)
        step_out = pipe.scheduler.step(noise_pred, t, latents, return_dict=True)
        return decode_image(pipe, step_out.pred_original_sample)

    img_stay = get_projection(proxy_current)
    img_switch = get_projection(proxy_next)

    w, h = img_stay.size
    combined = Image.new('RGB', (w * 2, h + 50), (20, 20, 20))
    combined.paste(img_stay, (0, 0))
    combined.paste(img_switch, (w, 0))

    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()

    draw.text((20, h + 5), f"Step {step_num}: LEFT (Stay)", fill="white", font=font)
    draw.text((w + 20, h + 5), f"Step {step_num}: RIGHT (Switch)", fill="cyan", font=font)

    return combined

# ==========================================
# 3. Main Agentic Generation Algorithm
# ==========================================

@torch.no_grad()
def generate_agentic(pipe, vlm_processor, vlm_model, prompts_list, seed=42):
    device = "cuda:0"
    generator = torch.Generator("cpu").manual_seed(seed)
    vlm_logs = []

    print(f"\n--- Starting AGENTIC Generation (Seed: {seed}) ---")
    
    pipe.scheduler.set_timesteps(50, device=device)
    timesteps = pipe.scheduler.timesteps
    
    print("Encoding prompts...")
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    
    encoded_proxies = []
    for p in prompts_list:
        (pe, ne, ppe, npe) = pipe.encode_prompt(prompt=p, negative_prompt="blur, noise", device=device)
        encoded_proxies.append({"pos": pe, "neg": ne, "pooled_pos": ppe, "pooled_neg": npe})
    
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

    latents = pipe.prepare_latents(
        1, pipe.unet.config.in_channels, 768, 768, 
        encoded_proxies[0]["pos"].dtype, device, generator
    )
    
    p_dim = int(pipe.text_encoder_2.config.projection_dim) if hasattr(pipe, "text_encoder_2") else 1280
    add_time_ids = pipe._get_add_time_ids((768, 768), (0, 0), (768, 768), latents.dtype, p_dim).to(device)
    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    current_idx = 0
    start_checking_step = 2
    check_interval = 2
    forced_switch_steps = [8, 12] 

    for i, t in enumerate(timesteps):
        if i % 10 == 0: print(f"Step {i+1}/50...")

        if current_idx < len(prompts_list) - 1:
            is_check_step = (i >= start_checking_step and i % check_interval == 0)
            is_timeout = (i >= forced_switch_steps[current_idx])

            if is_check_step or is_timeout:
                print(f"   [Lookahead] Step {i}: Generating Comparison...")
                proxy_curr = encoded_proxies[current_idx]
                proxy_next = encoded_proxies[current_idx + 1]
                
                comparison_img = create_phantom_comparison(
                    pipe, latents, t, proxy_curr, proxy_next, add_time_ids, i
                )
                
                comparison_path = f"debug_steps/step_{i}_seed_{seed}_decision.png"
                comparison_img.save(comparison_path)
                
                switch_triggered = False
                reason = ""

                if is_timeout:
                    print(f"     [!!!] TIMEOUT. Forced Switch.")
                    switch_triggered = True
                    reason = "timeout"
                else:
                    prompt_text = (
                        f"Look at this side-by-side comparison.\n"
                        f"LEFT: Current prompt\n"
                        f"RIGHT: Next prompt\n\n"
                        "Focus on the RIGHT image. Does it look like a valid variation of the Left image?\n"
                        "Ignore grain, blur, or noise. We only care if the general SHAPE and COMPOSITION are preserved.\n"
                        "If the Right side looks interesting and keeps the layout, say YES.\n"
                        "Answer with a single word: YES or NO."
                    )
                    
                    answer = query_vlm_dual_gpu(vlm_processor, vlm_model, comparison_img, question=prompt_text)
                    print(f"     -> Qwen says: '{answer}'")
                    
                    if "yes" in answer.lower():
                        switch_triggered = True
                        reason = answer

                if switch_triggered:
                    print(f"     [***] SWITCHING PROMPTS!")
                    current_idx += 1
                    vlm_logs.append({
                        "step": i, 
                        "event": "switch", 
                        "reason": reason, 
                        "phantom_path": comparison_path
                    })

        proxy = encoded_proxies[current_idx]
        noise_pred = pipe.unet(
            torch.cat([latents] * 2).to(device), t, 
            encoder_hidden_states=torch.cat([proxy["neg"], proxy["pos"]], dim=0),
            added_cond_kwargs={"text_embeds": torch.cat([proxy["pooled_neg"], proxy["pooled_pos"]], dim=0), "time_ids": add_time_ids},
            return_dict=False
        )[0]
        
        u, c = noise_pred.chunk(2)
        noise_pred = u + 7.5 * (c - u) 
        
        step_output = pipe.scheduler.step(noise_pred, t, latents, return_dict=True)
        latents = step_output.prev_sample

    return decode_image(pipe, latents), vlm_logs

# ==========================================
# 4. Main Execution Block
# ==========================================
if __name__ == "__main__":
    print(f"\n======================================================")
    print(f"AGENTIC SINGLE-IMAGE GENERATOR")
    print(f"========================================================\n")

    # 1. Parse the desired ID from command-line arguments (sys.argv)
    if len(sys.argv) < 2:
        print("[ERROR] Missing target ID.")
        print("Usage: python sdxl_single.py <target_id>")
        sys.exit(1)

    try:
        target_id = int(sys.argv[1])
    except ValueError:
        print(f"[ERROR] Invalid ID '{sys.argv[1]}'. Please provide a valid integer.")
        sys.exit(1)

    # 2. Load the JSON and locate the prompt
    try:
        with open('proxy_prompts.json', 'r') as file:
            prompt_dataset = json.load(file)
    except FileNotFoundError:
        print("\n[ERROR] 'proxy_prompts.json' not found in the current directory.")
        sys.exit(1)

    selected_item = next((item for item in prompt_dataset if item.get("id") == target_id), None)
    
    if not selected_item:
        print(f"\n[ERROR] Prompt ID '{target_id}' was not found in the JSON file.")
        sys.exit(1)

    safe_theme_name = selected_item['theme'].replace(" ", "_").lower()
    prompts = selected_item['prompts']
    current_seed = random.randint(1000, 999999) # Assign a random seed for unique generations

    print(f"\nTarget Found! Theme: {selected_item['theme']} | Seed: {current_seed}")

    # 3. Load Models
    t_main_start = time.time()
    vlm_processor, vlm_model = load_vlm()

    print("Loading SDXL to GPU 0...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda:0")
    
    pipe.vae.enable_tiling()
    pipe.vae = pipe.vae.to(dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # 4. Execute Agentic Generation
    img_agentic, logs = generate_agentic(
        pipe=pipe, vlm_processor=vlm_processor, vlm_model=vlm_model,
        prompts_list=prompts, seed=current_seed
    )
    
    # 5. Save outputs
    image_filename = f"results/agentic_768_{current_seed}_{safe_theme_name}.png"
    log_filename = f"results/agentic_logs_{current_seed}_{safe_theme_name}.json"
    
    img_agentic.save(image_filename)
    with open(log_filename, "w") as f:
        json.dump(logs, f, indent=4)
        
    print(f"\n=== Generation Complete! ===")
    print(f"Runtime: {(time.time() - t_main_start) / 60:.2f} minutes")
    print(f"Saved Image: {image_filename}")
    print(f"Saved Logs:  {log_filename}")