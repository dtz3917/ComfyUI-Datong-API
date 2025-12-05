import torch
from .processors.doubao_v3 import run_doubao, calculate_size_strategy

class Datong_API_Image:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "在此输入提示词..."}),
                
                # 1. 修改来源名称
                "api_provider": (["官方 (Volcengine)", "待更新..."], {"default": "官方 (Volcengine)"}),
                
                # 2. 这里的 Model ID 必须保留，因为火山引擎需要它
                # 我们把名字改得更清楚一点
                "model_ep_id": ("STRING", {"default": "", "multiline": False, "placeholder": "请去火山引擎后台复制你的 Endpoint ID (ep-xxxx)"}),
                
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "请填入你的 API Key"}),
                
                # 3. 比例选择
                "aspect_ratio": (["与原图一致 (Original)", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"], {"default": "与原图一致 (Original)"}),
                
                # 4. 分辨率修改为 1k 2k 4k
                "resolution": (["1k", "2k", "4k"], {"default": "2k"}),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "input_image_1": ("IMAGE",),
                "input_image_2": ("IMAGE",),
                "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",),
                "input_image_5": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "process_image"
    CATEGORY = "大桶API/Image"

    def process_image(self, prompt, api_provider, model_ep_id, api_key, aspect_ratio, resolution, seed, 
                      input_image_1=None, input_image_2=None, input_image_3=None, 
                      input_image_4=None, input_image_5=None):
        
        # 1. 整理图片
        images = [input_image_1, input_image_2, input_image_3, input_image_4, input_image_5]
        has_image = input_image_1 is not None

        # 2. 比例计算
        final_ratio = "1:1"
        if aspect_ratio == "与原图一致 (Original)" and has_image:
            _, h, w, _ = input_image_1.shape
            final_ratio = calculate_size_strategy(w, h, aspect_ratio)
            print(f"Auto-calculated Ratio: {final_ratio}")
        elif aspect_ratio == "与原图一致 (Original)" and not has_image:
            final_ratio = "1:1"
        else:
            final_ratio = aspect_ratio

        # 3. 路由分发
        result_image = None
        
        if api_provider == "官方 (Volcengine)":
            # 如果你有特定的 prompt 增强需求，可以在这里把 final_ratio 加进去
            # 比如: prompt = f"{prompt}, aspect ratio {final_ratio}"
            
            result_image = run_doubao(api_key, prompt, images, resolution, seed, model_ep_id)
        
        else:
            print(f"Provider {api_provider} not ready.")
            result_image = (torch.zeros((1, 512, 512, 3)),)

        # 错误兜底：如果 API 任何原因没返回图，给一张纯黑图提示用户，而不是直接让 ComfyUI 红框崩溃
        if result_image is None or result_image[0] is None:
             print("Generation failed. Returning empty image.")
             return (torch.zeros((1, 512, 512, 3)),)

        return result_image