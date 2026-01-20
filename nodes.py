# develop code way~
"""
    cd comfyui & ipython
    `
    import torch

    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()
    from nodes import NODE_CLASS_MAPPINGS
    import_custom_nodes()

    loadimage = NODE_CLASS_MAPPINGS["LoadImage"]().load_image()
    self_define_node import or paster and ...
    ...
    `
"""
import sys
import os


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import torch
from typing import Union, List

from PIL import Image, ImageDraw, ImageFont, ImageChops
# from comfy.model_patcher import ModelPatcher
print(">> zdx_comfyui: starting local imports...")
try:
    print(">> zdx_comfyui: imports done.")
except Exception as e:
    print(">> zdx_comfyui load ERROR:", e)

# InstantFaceSwap = None
# try:
#     from .annotator import FaceDetector
#     from .face_reactor_instantid import InstantFaceSwap
# except Exception as e:
#     print(">> zdx_comfyui InstantFaceSwap load ERROR:", e)
# try:
#     from .llm  import QwenClient
# except Exception as e:
#     print(">> zdx_comfyui QwenClient load ERROR:", e)

# print(">> zdx_comfyui: importing layerstyle mappings...")
# from .layerstyle import NODE_CLASS_MAPPINGS as layerstyle_NODE_CLASS_MAPPINGS
# from .layerstyle import NODE_DISPLAY_NAME_MAPPINGS as layerstyle_NODE_DISPLAY_NAME_MAPPINGS
# print(">> zdx_comfyui: layerstyle mappings imported.")



def pil2tensor(image):
    new_image = image.convert('RGB')
    img_array = np.array(new_image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)[None]
    return img_tensor
def tensor2pil(image):
    if len(image.shape) < 3:
        image = image.unsqueeze(0)
    return Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))

def pilmask2tensor(mask_img):
    mask_tensor = torch.from_numpy(np.array(mask_img.convert('L'))).float()  # 转换为float类型
    mask_tensor = mask_tensor / 255.0  # 归一化到 0-1 范围
    mask_tensor = mask_tensor.unsqueeze(0)
    return mask_tensor

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

def kjtensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]

def kjpil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([kjpil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def image2mask(image: Image.Image) -> torch.Tensor:
    if isinstance(image, Image.Image):
        image = pil2tensor(image)
    return image.squeeze()[..., 0]

def mask2image(mask: torch.Tensor) -> Image.Image:
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0)
    return tensor2pil(mask)

def RGB2RGBA(image: Image.Image, mask: Union[Image.Image, torch.Tensor]) -> Image.Image:
    if isinstance(mask, torch.Tensor):
        mask = mask2image(mask)
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    return Image.merge('RGBA', (*image.convert('RGB').split(), mask.convert('L')))


def draw_text(image, text, position=(50, 50), font_size=45, color=(255, 255, 255)):  # 默认白色
    draw = ImageDraw.Draw(image)
    # 根据图像模式选择适当的颜色格式
    if image.mode == 'RGB':
        color = (255, 255, 255) if color == 255 else color  # RGB模式
    elif image.mode == 'RGBA':
        color = (255, 255, 255, 255) if color == 255 else color  # RGBA模式
    elif image.mode in ['L', '1']:
        color = 255 if isinstance(color, tuple) else color  # 灰度图模式
    
    font = ImageFont.load_default(size=font_size)
    
    # 绘制文本
    draw.text(position, text, font=font, fill=color)
    
    return image

class NoneNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {}
        }
    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("ANY", "MASK", "boolean")
    FUNCTION = "none"
    CATEGORY = "zdx/logic"
    def none(self):
        return (None, None, False)

class isEmpty:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"mask": ("MASK",),},
            "optional": {}
        }
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = "zdx/Logic"
    def execute(self, mask):
        if mask is None:
            return (True,)
        return (False,)

class SizeNormalizer():
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "image": ("IMAGE",),
                        "max_size": ("INT", { "default": 2048, "min": 1024, "max": 4096, "step": 1, }),
                    }
                }

    CATEGORY = "zdx/image"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)

    FUNCTION = "normal_size"

    def normal_size(self, image, max_size):
        image = tensor2pil(image)
        w, h = image.size
        if w * h > max_size * max_size or w > max_size or h > max_size:
            # Calculate new dimensions while maintaining aspect ratio
            if w > h:
                new_w = max_size
                new_h = int(h * (max_size / w))
            else:
                new_h = max_size
                new_w = int(w * (max_size / h))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        image = pil2tensor(image)
        return (image,)

class zdxApplySageAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "use_SageAttention": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "zdx"
    TITLE = "zdx SageAttention"

    def __init__(self):
        self.orig_attn = None

    # def patch(self, model: ModelPatcher, use_SageAttention: bool):
    def patch(self, model, use_SageAttention):
        try:
            from comfy.ldm.flux import math

            if use_SageAttention:
                from sageattention import sageattn
                from comfy.ldm.modules.attention import attention_sage
                from comfy.ldm.modules import attention

                self.orig_attn = getattr(math, "optimized_attention")
                setattr(attention, "sageattn", sageattn)
                setattr(math, "optimized_attention", attention_sage)
            elif self.orig_attn is not None:
                setattr(math, "optimized_attention", self.orig_attn)
        except:
            pass
        return (model,)

# Register the node
NODE_CLASS_MAPPINGS = {
    "NoneNode": NoneNode,
    "SizeNormalizer": SizeNormalizer,
    "zdxApplySageAttention": zdxApplySageAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoneNode": "NoneNode",
    "SizeNormalizer": "image size normalizer",
    "zdxApplySageAttention": "zdxApplySageAttention",
}

print(">> zdx_comfyui: local imports done.")




# Add custom API routes, using router
# from aiohttp import web
# from server import PromptServer
