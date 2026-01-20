# ComfyUI-RMBG
# This custom node for ComfyUI provides functionality for background removal using various models,
# including RMBG-2.0, INSPYRENET, and BEN. It leverages deep learning techniques
# to process images and generate masks for background removal.
# 
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/AILab-AI/ComfyUI-RMBG

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
from PIL import Image, ImageFilter
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import folder_paths
from huggingface_hub import hf_hub_download
import shutil
from torchvision import transforms

import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

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

device = "cuda" if torch.cuda.is_available() else "cpu"

folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

AVAILABLE_MODELS = {
    "segformer_b2_clothes": "1038lab/segformer_clothes"
}

class ClothesSegment:
    def __init__(self):
        self.processor = None
        self.model = None
        self.cache_dir = os.path.join(folder_paths.models_dir, "RMBG", "segformer_clothes")
    
    @classmethod
    def INPUT_TYPES(cls):
        available_classes = ["Hat", "Hair", "Face", "Sunglasses", "Upper-clothes", "Skirt", "Dress", "Belt", "Pants", "Left-arm", "Right-arm", "Left-leg", "Right-leg", "Bag", "Scarf", "Left-shoe", "Right-shoe","Background"]
        
        tooltips = {
            "process_res": "Processing resolution (higher = more VRAM)",
            "mask_blur": "Blur amount for mask edges",
            "mask_offset": "Expand/Shrink mask boundary",
            "invert_output": "Invert both image and mask output",
            "background": "Choose background type: Alpha (transparent) or Color (custom background color).",
            "background_color": "Choose background color (Alpha = transparent)"
        }
        
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                **{cls_name: ("BOOLEAN", {"default": False}) 
                   for cls_name in available_classes},
                "process_res": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 32, "tooltip": tooltips["process_res"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": tooltips["background"]}),
                "background_color": ("COLORCODE", {"default": "#222222", "tooltip": tooltips["background_color"]}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "segment_clothes"
    CATEGORY = "zdx/🧽RMBG"

    def check_model_cache(self):
        if not os.path.exists(self.cache_dir):
            return False, "Model directory not found"
        
        required_files = [
            'config.json',
            'model.safetensors',
            'preprocessor_config.json'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.cache_dir, f))]
        if missing_files:
            return False, f"Required model files missing: {', '.join(missing_files)}"
        return True, "Model cache verified"

    def clear_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()

    def download_model_files(self):
        model_id = AVAILABLE_MODELS["segformer_b2_clothes"]
        model_files = {
            'config.json': 'config.json',
            'model.safetensors': 'model.safetensors',
            'preprocessor_config.json': 'preprocessor_config.json'
        }
        
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Downloading Clothes Segformer model files...")
        
        try:
            for save_name, repo_path in model_files.items():
                print(f"Downloading {save_name}...")
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=repo_path,
                    local_dir=self.cache_dir,
                    local_dir_use_symlinks=False
                )
                
                if os.path.dirname(downloaded_path) != self.cache_dir:
                    target_path = os.path.join(self.cache_dir, save_name)
                    shutil.move(downloaded_path, target_path)
            return True, "Model files downloaded successfully"
        except Exception as e:
            return False, f"Error downloading model files: {str(e)}"

    def segment_clothes(self, images, process_res=1024, mask_blur=0, mask_offset=0, background="Alpha", background_color="#222222", invert_output=False, **class_selections):
        try:
            # Check and download model if needed
            cache_status, message = self.check_model_cache()
            if not cache_status:
                print(f"Cache check: {message}")
                download_status, download_message = self.download_model_files()
                if not download_status:
                    raise RuntimeError(download_message)
            
            # Load model if needed
            if self.processor is None:
                # self.processor = SegformerImageProcessor.from_pretrained(self.cache_dir) # Not needed for direct tensor processing
                self.model = AutoModelForSemanticSegmentation.from_pretrained(self.cache_dir)
                self.model.eval()
                # for param in self.model.parameters():
                #     param.requires_grad = False
                self.model.to(device)

            # Class mapping for segmentation
            class_map = {
                "Background": 0, "Hat": 1, "Hair": 2, "Sunglasses": 3, 
                "Upper-clothes": 4, "Skirt": 5, "Pants": 6, "Dress": 7,
                "Belt": 8, "Left-shoe": 9, "Right-shoe": 10, "Face": 11,
                "Left-leg": 12, "Right-leg": 13, "Left-arm": 14, "Right-arm": 15,
                "Bag": 16, "Scarf": 17
            }

            # Get selected classes
            selected_classes = [name for name, selected in class_selections.items() if selected]
            if not selected_classes:
                selected_classes = ["Upper-clothes"]

            # Batch processing setup
            B, H, W, C = images.shape
            process_device = device
            images_device = images.to(process_device) # (B, H, W, C)

            # 1. Preprocess (Torch/GPU)
            # Permute to (B, C, H, W)
            x = images_device.permute(0, 3, 1, 2)
            
            # Resize to process_res
            if H != process_res or W != process_res:
                x_resized = F.interpolate(x, size=(process_res, process_res), mode='bilinear', align_corners=False)
            else:
                x_resized = x
            
            # Normalize (ImageNet mean/std)
            # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
            mean = torch.tensor([0.485, 0.456, 0.406], device=process_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=process_device).view(1, 3, 1, 1)
            
            input_tensor = (x_resized - mean) / std

            # 2. Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits # (B, num_classes, H_out, W_out)
                
                # 3. Post-process
                # Upsample logits to original size
                # Process in chunks to avoid INT_MAX limitation in interpolate and OOM
                pred_seg_list = []
                # Use small chunk size for upsampling large tensors
                interp_chunk_size = 1 
                
                for i in range(0, B, interp_chunk_size):
                    end_idx = min(i + interp_chunk_size, B)
                    chunk_logits = logits[i:end_idx]
                    
                    chunk_upsampled = F.interpolate(
                        chunk_logits,
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    )
                    # Argmax immediately to save memory
                    chunk_pred = chunk_upsampled.argmax(dim=1) # (chunk_size, H, W)
                    pred_seg_list.append(chunk_pred)
                
                pred_seg = torch.cat(pred_seg_list, dim=0) # (B, H, W)
                
                # Combine masks
                combined_mask = torch.zeros((B, H, W), dtype=torch.float32, device=process_device)
                
                for class_name in selected_classes:
                    idx = class_map[class_name]
                    combined_mask += (pred_seg == idx).float()
                
                combined_mask = torch.clamp(combined_mask, 0, 1)
                
                # Add channel dim for processing: (B, 1, H, W)
                combined_mask = combined_mask.unsqueeze(1)
                
                # Blur
                if mask_blur > 0:
                    k_size = mask_blur * 2 + 1
                    combined_mask = gaussian_blur(combined_mask, kernel_size=k_size)
                
                # Offset
                if mask_offset != 0:
                    if mask_offset > 0:
                        k_size = mask_offset * 2 + 1
                        combined_mask = F.max_pool2d(combined_mask, kernel_size=k_size, stride=1, padding=mask_offset)
                    else:
                        k_size = -mask_offset * 2 + 1
                        combined_mask = -F.max_pool2d(-combined_mask, kernel_size=k_size, stride=1, padding=-mask_offset)

                if invert_output:
                    combined_mask = 1.0 - combined_mask
                
                # Final mask: (B, H, W, 1)
                mask_final = combined_mask.permute(0, 2, 3, 1)

                # Composite
                if background == "Alpha":
                    result_images = torch.cat([images_device, mask_final], dim=-1)
                else:
                    hex_color = background_color.lstrip('#')
                    if len(hex_color) == 6:
                        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                    elif len(hex_color) == 8:
                        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                    else:
                        r, g, b = 0, 0, 0
                    
                    bg_tensor = torch.tensor([r, g, b], dtype=torch.float32, device=process_device).view(1, 1, 1, 3) / 255.0
                    result_images = images_device * mask_final + bg_tensor * (1.0 - mask_final)

            return (result_images.cpu(), mask_final.squeeze(-1).cpu(), mask_final.repeat(1, 1, 1, 3).cpu())

        except Exception as e:
            self.clear_model()
            raise RuntimeError(f"Error in Clothes Segformer processing: {str(e)}")
        finally:
            self.clear_model()

NODE_CLASS_MAPPINGS = {
    "zClothesSegment": ClothesSegment
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "zClothesSegment": "Clothes Segment (zRMBG)"
} 