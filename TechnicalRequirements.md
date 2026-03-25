# Technical Requirements

This document outlines the hardware, software, and storage prerequisites necessary to run the **Dynamic Stage-Aware Prompt Injection** pipeline. Because this architecture runs a large diffusion model (SDXL) and a Vision-Language Model (Qwen2-VL) concurrently, a multi-GPU environment is highly recommended.

---

## 🖥️ Hardware Requirements

### 1. GPU (Graphics Processing Unit)
The provided scripts (`sdxl.py` and `sdxl_single.py`) are strictly designed for a **Dual-GPU architecture** to prevent Out-Of-Memory (OOM) errors and computational bottlenecks.
* **GPU 0 (Diffusion - SDXL):** Minimum **11 GB VRAM** (Successfully tested on NVIDIA RTX 2080 Ti). 
* **GPU 1 (Agentic Decision - Qwen2-VL-2B):** Minimum **8 GB VRAM** (Successfully tested on NVIDIA RTX 2080 Ti). 
* *Note:* If you only have a single high-capacity GPU (e.g., a single 24GB RTX 3090 or 40GB A100), you must modify the python scripts to map both the `pipe` and the VLM to `"cuda:0"`.

### 2. System Memory (RAM)
* **Minimum:** 32 GB DDR4
* **Recommended:** 64 GB (Ideal for caching large model weights into RAM during initial loading phases).

### 3. Storage & Quotas (Crucial for University/Enterprise Clusters)
* **Disk Space:** At least **30 GB of free space** on a fast SSD. 
* *Breakdown:* SDXL Base 1.0 requires ~14GB, and Qwen2-VL-2B-Instruct requires ~5GB. The remaining space is needed for PyTorch wheels, intermediate latent decoding overhead, and saving the generated image grids.
* **⚠️ Cluster Quota Warning:** If running on a shared cluster (like Slurm), **do not run this in a restricted `~/.home` directory** if your account has a small storage quota (e.g., 4GB). The Hugging Face model downloads and `pip` caches will instantly max out your quota and crash the OS. Always clone and run this repository in your assigned `/scratch` or `/work` filesystem.

---

## ⚙️ Software Requirements

### 1. Operating System
* **Linux:** Ubuntu 20.04 or 22.04 LTS (Native environment for cluster/Slurm execution).
* **Windows:** Windows 11 with WSL2 (Windows Subsystem for Linux) configured with the NVIDIA Container Toolkit.

### 2. Core Dependencies
* **Python:** Version `3.10.x`
* **CUDA Toolkit:** Version `11.8` or `12.1` (Must match your PyTorch installation).
* **Conda:** Miniconda or Anaconda for environment management.

### 3. Hugging Face Authentication & Caching
The script downloads models directly from the Hugging Face Hub. 
* By default, the script enforces a local cache directory to avoid filling up hidden system partitions: `os.environ["HF_HOME"] = "./.hidden_cache"`. Ensure you have read/write permissions in the execution directory.

---

## 📦 Python Environment Setup

We recommend using `conda` to ensure isolated dependencies and correct C++ compiler bindings for PyTorch.

### 1. Create the Environment
```bash
conda create -n dynamic_prompt_env python=3.10 -y
conda activate dynamic_prompt_env