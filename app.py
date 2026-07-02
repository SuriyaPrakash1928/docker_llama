# import os
# from llama_cpp import Llama

# # os.environ["GGML_VULKAN_DEVICE"] = "1"

# MODEL_PATH = "./models/qwen2.5.gguf"

# print("Loading AI model into memory... Please wait.")
# # n_ctx=2048 sets the memory context window. 
# llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, tensor_split=[0.10, 0.90], n_ctx=2048, split_mode=2, verbose=True)


# def chat(prompt):

#     user_message = prompt
    
#     # Generate the AI response
#     response = llm.create_chat_completion(
#         messages=[
#             {"role": "user", "content": user_message}
#         ],
#         max_tokens=8000,
#         temperature=0.7,
#         stream= True,
#     )
    
#     full_response = ""
#     for chunk in response:
#         delta = chunk['choices'][0]['delta']
#         if 'content' in delta:
#             token = delta['content']
#             print(token, end='', flush=True)
#             full_response += token
    
#     print()
#     return full_response

# print("Local AI Chatbot")
# print("Powered by llama.cpp & Qwen")
# while True:
#     user_message = input("Enter the prompt here...\n")
#     if (user_message.lower() in ['exit','bye','goodbye']):
#         print("Bye! If you need any further assistance feel free to ask.")
#         break
#     else:
#         chat(user_message)

import os
import platform
import subprocess
import inspect
from llama_cpp import Llama

# 1. Import pynvml for NVIDIA hardware detection
try:
    import pynvml
    PYNVML_AVAILABLE = True
    print("PYNVML is Available")
except ImportError:
    PYNVML_AVAILABLE = False
    print("PYNVML is not Available")

def detect_nvidia_gpus():
    """Detects NVIDIA GPUs and their VRAM using NVML."""
    nvidia_gpus = []
    if not PYNVML_AVAILABLE:
        print("Warning: 'pynvml' not found. Install it using: pip install pynvml")
        return nvidia_gpus

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            vram_gb = info.total / (1024**3)
            nvidia_gpus.append({
                "id": i, 
                "name": name, 
                "vram_gb": round(vram_gb, 2),
                "backend": "CUDA"
            })
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"NVML hardware detection failed: {e}")
        
    return nvidia_gpus

def detect_intel_gpus():
    """Detects Intel GPUs using system commands (Windows wmic)."""
    intel_gpus = []
    if platform.system() == "Windows":
        try:
            # Use wmic to get video controller names (suppresses deprecation warnings)
            output = subprocess.check_output(
                ['wmic', 'path', 'win32_videocontroller', 'get', 'name'], 
                text=True, 
                stderr=subprocess.DEVNULL
            )
            for line in output.strip().split('\n')[1:]:
                if "Intel" in line and line.strip():
                    intel_gpus.append({"name": line.strip(), "backend": "Vulkan/OpenCL"})
        except Exception:
            pass
    return intel_gpus

def analyze_and_calculate_split(nvidia_gpus, intel_gpus):
    """
    Compares GPUs, determines the best one, and calculates tensor_split.
    """
    print("\n--- AI Accelerator Analysis ---")
    
    # Print all detected hardware
    for gpu in nvidia_gpus:
        print(f"[DETECTED] {gpu['name']} | VRAM: {gpu['vram_gb']} GB | Backend: {gpu['backend']}")
    for gpu in intel_gpus:
        print(f"[DETECTED] {gpu['name']} | Backend: {gpu['backend']} (Shared System RAM)")
        
    # Determine primary AI accelerator
    if not nvidia_gpus:
        print("\n[DECISION] No NVIDIA GPUs found. AI will run on CPU (Very Slow).")
        return None
        
    best_gpu = max(nvidia_gpus, key=lambda x: x["vram_gb"])
    print(f"\n[DECISION] Primary AI Accelerator: {best_gpu['name']}")
    print("Reason: llama-cpp-python requires CUDA for GPU acceleration. Intel Iris Xe uses Vulkan/OpenCL.")
    
    if intel_gpus:
        print(f"[NOTE] {intel_gpus[0]['name']} detected, but CANNOT be used in tensor_split.")
        print("Reason: llama.cpp cannot split tensors across different backends (CUDA vs Vulkan).")
        
    # Calculate tensor_split for NVIDIA GPUs only
    if len(nvidia_gpus) == 1:
        print(f"\n[CALCULATION] Only 1 NVIDIA GPU available. tensor_split is not needed.")
        return None
    else:
        total_vram = sum(gpu["vram_gb"] for gpu in nvidia_gpus)
        tensor_split = [gpu["vram_gb"] / total_vram for gpu in nvidia_gpus]
        print(f"\n[CALCULATION] Multiple NVIDIA GPUs detected. Auto-calculated tensor_split: {tensor_split}")
        return tensor_split

# --- Main Execution ---
nvidia_gpus = detect_nvidia_gpus()
intel_gpus = detect_intel_gpus()

tensor_split_ratio = analyze_and_calculate_split(nvidia_gpus, intel_gpus)

MODEL_PATH = "./models/qwen2.5.gguf"
print("\nLoading AI model into memory... Please wait.")

# Build initialization arguments dynamically
llama_kwargs = {
    "model_path": MODEL_PATH,
    "n_gpu_layers": -1,  # Offload all layers to the available NVIDIA GPU(s)
    "n_ctx": 2048,
    "verbose": False
}

# Add tensor_split if multiple NVIDIA GPUs exist
if tensor_split_ratio is not None:
    llama_kwargs["tensor_split"] = tensor_split_ratio

# Dynamically check if the installed llama-cpp-python version supports 'split_mode'
# (It was removed in v0.2.x+, but older versions require it)
if 'split_mode' in inspect.signature(Llama).parameters:
    llama_kwargs["split_mode"] = 2  # 2 = LLAMA_SPLIT_MODE_ROW
    print("[CONFIG] Legacy split_mode=2 enabled for your llama-cpp-python version.")

llm = Llama(**llama_kwargs)

def chat(prompt):
    user_message = prompt
    
    # Generate the AI response
    response = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": user_message}
        ],
        max_tokens=8000,
        temperature=0.7,
        stream=True,
    )
    
    full_response = ""
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            token = delta['content']
            print(token, end='', flush=True)
            full_response += token
    
    print() # Print a newline after the response finishes
    return full_response

print("\nLocal AI Chatbot")
print("Powered by llama.cpp & Qwen")
while True:
    try:
        user_message = input("Enter the prompt here...\n")
        if (user_message.lower() in ['exit','bye','goodbye']):
            print("Bye! If you need any further assistance feel free to ask.")
            break
        else:
            chat(user_message)
    except KeyboardInterrupt:
        # Allows the user to press Ctrl+C to exit gracefully
        print("\nBye! If you need any further assistance feel free to ask.")
        break