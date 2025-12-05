import requests
import json
import base64
import io
from PIL import Image
import torch
import numpy as np

# 辅助函数：把 ComfyUI 的 Tensor 图片转换为 Base64 字符串
def tensor_to_base64(image_tensor):
    if image_tensor is None:
        return None
    # ComfyUI 图片是 [Batch, H, W, C]，我们取第一张
    i = 255. * image_tensor[0].cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    
    # 【核心修复】: 之前是 PNG，体积太大导致报错。
    # 改为 JPEG 格式，quality=95 (高质量压缩)，体积通常能减少 90%
    # JPEG 不支持透明通道(RGBA)，所以必须先转 RGB
    img = img.convert("RGB")
    
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # 注意这里前缀变成了 image/jpeg
    return f"data:image/jpeg;base64,{img_str}"

# 辅助函数：自动计算比例
def calculate_size_strategy(width, height, target_ratio_str):
    ratios = {
        "1:1": 1.0, "4:3": 1.33, "3:4": 0.75,
        "16:9": 1.77, "9:16": 0.56, "3:2": 1.5, "2:3": 0.66
    }
    
    if target_ratio_str == "与原图一致 (Original)":
        current_ratio = width / height
        closest_ratio = min(ratios.keys(), key=lambda x: abs(ratios[x] - current_ratio))
        return closest_ratio
    
    return target_ratio_str

# 核心：豆包 API 调用函数
def run_doubao(api_key, prompt, images, size_selection, seed, model_id):
    url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 保持之前的修复：Seed 限制在 21 亿以内
    safe_seed = seed % 2147483647

    api_size = size_selection.upper()

    payload = {
        "model": model_id,
        "prompt": prompt,
        "sequential_image_generation": "disabled",
        "response_format": "b64_json", 
        "size": api_size, 
        "stream": False,
        "watermark": False,
        "seed": safe_seed 
    }

    valid_images = [img for img in images if img is not None]
    
    if len(valid_images) > 0:
        base64_images = [tensor_to_base64(img) for img in valid_images]
        if len(base64_images) == 1:
            payload["image"] = base64_images[0]
        else:
            payload["image"] = base64_images

    print(f"Make Request to Doubao... Model: {model_id} | Seed: {safe_seed} | Size: {api_size}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                img_data = data["data"][0].get("b64_json") or data["data"][0].get("url")
                
                if img_data.startswith("http"):
                    img_resp = requests.get(img_data)
                    image = Image.open(io.BytesIO(img_resp.content))
                else:
                    if "," in img_data:
                        img_data = img_data.split(",")[1]
                    image = Image.open(io.BytesIO(base64.b64decode(img_data)))
                
                image = image.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                return (image,)
            else:
                print("Error: No data in response")
                return (None,)
        else:
            print(f"API Error Code: {response.status_code}")
            print(f"API Error Body: {response.text}")
            return (None,)
            
    except Exception as e:
        print(f"Exception during API call: {e}")
        return (None,)
