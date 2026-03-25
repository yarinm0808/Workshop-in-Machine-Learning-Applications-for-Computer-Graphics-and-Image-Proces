# Dynamic Stage-Aware Prompt Injection for Contextually-Contradictory Image Generation

[cite_start]**Workshop in ML Applications for Computer Graphics and Image Processing - Tel Aviv University** [cite: 2, 3]

[cite_start]**Authors:** Yarin Meirovich, Tore Barach Kamar, Shahar Ghivoly [cite: 3]

## 📖 Overview
[cite_start]Diffusion models have become the de-facto standard for high-fidelity image generation, yet they consistently struggle to generate semantically accurate results when prompts feature "contextual contradictions" (concepts rarely seen together in the model's training distribution)[cite: 6]. 

[cite_start]While prior solutions attempted to solve this by using Large Language Models (LLMs) to statically schedule prompt-switching during the diffusion process, these static heuristics fail to account for the variance between different generation seeds, model architectures, and denoising schedules[cite: 7, 8]. 

[cite_start]This project introduces a **dynamic, stage-aware prompt injection framework**[cite: 9]. [cite_start]By running a Vision-Language Model (VLM) concurrently with the diffusion process, our system evaluates "lookahead" previews of the denoised latent space to dynamically trigger prompt switches on a per-seed, per-step basis[cite: 10, 76].

## ⚠️ The Problem: Concept Entanglement & Static Failures
[cite_start]When a model is asked to generate implausible associations like *"A cruise ship parked in a bathtub"* or *"A monkey juggles tiny elephants"*, the cross-attention mechanism struggles to isolate the structural representation of the primary subject from its highly correlated environment[cite: 13, 14, 39]. [cite_start]The concepts become deeply "entangled" in the model's learned priors, resulting in omitted elements or unrecognizable blending[cite: 38, 41].

[cite_start]The baseline solution of static prompt injection (splitting a prompt into proxy prompts and switching at hard-coded steps) suffers from a formidable flaw[cite: 50, 60]:
* [cite_start]**Switching too early:** The model loses the structural pose[cite: 63].
* [cite_start]**Switching too late:** The new attributes fail to materialize fully[cite: 64].
* [cite_start]**The Root Cause:** The optimal step to introduce a new element varies wildly depending on the initial random noise seed and the technical details of the model itself[cite: 61, 62].

## 🧠 Our Solution: The Lookahead Mechanism
[cite_start]To prevent harsh text-embedding switches from irreparably fracturing the latent space, our pipeline queries the VLM with side-by-side potential futures[cite: 81, 82]. 

[cite_start]After completing an adjustable interval of denoising steps, the pipeline temporarily branches to simulate future steps under two scenarios[cite: 83, 84]:
1. [cite_start]**The "Stay" Path (Left):** What the next steps look like if we do not switch the prompt[cite: 85].
2. [cite_start]**The "Switch" Path (Right):** What the next steps look like if we do switch the prompt[cite: 86].

[cite_start]The VLM evaluates both generated futures and will only commit to the "Switch" path if it attests that the visual structure remains stable while the new attributes begin to appear[cite: 87, 88].

## 🗂️ Repository Structure

This repository contains two primary generation scripts, tailored for different use cases (batch validation vs. single targeted generation), alongside configuration files.

* `sdxl.py` - **The Validation Suite:** Runs the full automated validation loop. It iterates through all 20 prompts in the JSON file across multiple random seeds, generating *both* the dynamic Agentic image and the static Baseline image for side-by-side comparison.
* `sdxl_single.py` - **The Targeted Generator:** A streamlined script for generating a single image using only the dynamic Agentic method. It takes a prompt ID via the command line, looks it up in the JSON, and generates a unique image using a random seed.
* `proxy_prompts.json` - **The Configuration File:** Contains the dataset of 20 challenging contextual contradiction themes, complete with their sequential proxy prompts and static baseline switch steps.
* `TECHNICAL_REQUIREMENTS.md` - Hardware and system prerequisites for the dual-GPU setup.
* `requirements.txt` - Python package dependencies.

## ⚙️ Technical Architecture
[cite_start]Our implementation utilizes a concurrent, dual-GPU pipeline to mitigate the computational bottleneck of running large models simultaneously[cite: 97].
* [cite_start]**GPU 0 (SDXL - Diffusion Generation):** Handles the standard UNet diffusion process and maintains the Text Encoder[cite: 98, 99]. [cite_start]The inference operator code is retrofitted with "breakpoints" to enable prompt injection and inspection of predicted futures[cite: 100]. [cite_start]At predefined intervals, noisy latents are extracted and passed through a lightweight Decoder to create a preview image[cite: 119].
* [cite_start]**GPU 1 (Qwen2-VL - Agentic Decision):** Hosts the VLM responsible for evaluating the visual state of the generation[cite: 101]. [cite_start]It receives the preview image and a targeted question[cite: 120]. [cite_start]If it outputs a "YES" decision, a trigger is sent back to GPU 0 to execute the prompt switch[cite: 121].

## 💻 Installation & Usage

**1. Environment Setup**
```bash
# Clone the repository
git clone [https://github.com/yourusername/dynamic-prompt-injection.git](https://github.com/yourusername/dynamic-prompt-injection.git)
cd dynamic-prompt-injection

# Set up the conda environment
conda create -n dynamic_prompt_env python=3.10
conda activate dynamic_prompt_env
pip install -r requirements.txt