import os
from llama_cpp import Llama

# os.environ["GGML_VULKAN_DEVICE"] = "1"

MODEL_PATH = "./models/qwen2.5.gguf"

print("Loading AI model into memory... Please wait.")
# n_ctx=2048 sets the memory context window. 
llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, tensor_split=[0.10, 0.90], n_ctx=2048, split_mode=2, verbose=True)


def chat(prompt):

    user_message = prompt
    
    # Generate the AI response
    response = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": user_message}
        ],
        max_tokens=8000,
        temperature=0.7,
        stream= True,
    )
    
    full_response = ""
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            token = delta['content']
            print(token, end='', flush=True)
            full_response += token
    
    print()
    return full_response

print("Local AI Chatbot")
print("Powered by llama.cpp & Qwen")
while True:
    user_message = input("Enter the prompt here...\n")
    if (user_message.lower() in ['exit','bye','goodbye']):
        print("Bye! If you need any further assistance feel free to ask.")
        break
    else:
        chat(user_message)