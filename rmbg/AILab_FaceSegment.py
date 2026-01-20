# ComfyUI-RMBG
# This custom node for ComfyUI provides functionality for face parsing using Segformer model.
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
    "face_parsing": "1038lab/segformer_face"
}

class FaceSegment:
    def __init__(self):
        self.processor = None
        self.model = None
        self.cache_dir = os.path.join(folder_paths.models_dir, "RMBG", "segformer_face")
    
    @classmethod
    def INPUT_TYPES(cls):
        available_classes = [
            "Skin", "Nose", "Eyeglasses", "Left-eye", "Right-eye",
            "Left-eyebrow", "Right-eyebrow", "Left-ear", "Right-ear", "Mouth",
            "Upper-lip", "Lower-lip", "Hair", "Earring", "Neck",
        ]
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
    FUNCTION = "segment_face"
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
        model_id = AVAILABLE_MODELS["face_parsing"]
        model_files = {
            'config.json': 'config.json',
            'model.safetensors': 'model.safetensors',
            'preprocessor_config.json': 'preprocessor_config.json'
        }
        
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Downloading face parsing model files...")
        
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

    def segment_face(self, images, process_res=512, mask_blur=0, mask_offset=0, background="Alpha", background_color="#222222", invert_output=False, **class_selections):
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
                self.processor = SegformerImageProcessor.from_pretrained(self.cache_dir)
                self.model = AutoModelForSemanticSegmentation.from_pretrained(self.cache_dir)
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.to(device)

            # Class mapping for segmentation
            class_map = {
                "Background": 0, "Skin": 1, "Nose": 2, "Eyeglasses": 3,
                "Left-eye": 4, "Right-eye": 5, "Left-eyebrow": 6, "Right-eyebrow": 7,
                "Left-ear": 8, "Right-ear": 9, "Mouth": 10, "Upper-lip": 11,
                "Lower-lip": 12, "Hair": 13, "Hat": 14, "Earring": 15,
                "Necklace": 16, "Neck": 17, "Clothing": 18
            }

            # Get selected classes
            selected_classes = [name for name, selected in class_selections.items() if selected]
            if not selected_classes:
                selected_classes = ["Skin", "Nose", "Left-eye", "Right-eye", "Mouth"]
            
            # Validate selected classes
            invalid_classes = [cls for cls in selected_classes if cls not in class_map]
            if invalid_classes:
                raise ValueError(f"Invalid class selections: {', '.join(invalid_classes)}. Valid classes are: {', '.join(class_map.keys())}")

            # Image preprocessing (Batch GPU)
            B, H, W, C = images.shape
            
            # Prepare constants
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            
            if background != "Alpha":
                hex_color = background_color.lstrip('#')
                if len(hex_color) == 6:
                    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                elif len(hex_color) == 8:
                    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                else:
                    r, g, b = 0, 0, 0
                bg_tensor = torch.tensor([r, g, b], dtype=torch.float32, device=device).view(1, 1, 1, 3) / 255.0

            result_images_list = []
            mask_final_list = []
            
            # Process in mini-batches to avoid OOM
            mini_batch_size = 8
            
            for i in range(0, B, mini_batch_size):
                end_idx = min(i + mini_batch_size, B)
                batch_imgs = images[i:end_idx].to(device)
                current_B = batch_imgs.shape[0]
                
                # Permute to (B, C, H, W)
                x = batch_imgs.permute(0, 3, 1, 2)
                
                # Resize
                if H != process_res or W != process_res:
                    x_resized = F.interpolate(x, size=(process_res, process_res), mode='bilinear', align_corners=False)
                else:
                    x_resized = x
                
                input_tensor = (x_resized - mean) / std

                # Inference
                with torch.inference_mode():
                    outputs = self.model(input_tensor)
                    logits = outputs.logits # (current_B, 19, res, res)

                    # Post-process
                    upsampled_logits = F.interpolate(
                        logits,
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred_seg = upsampled_logits.argmax(dim=1) # (current_B, H, W)

                # Combine masks (GPU)
                combined_mask = torch.zeros((current_B, H, W), dtype=torch.float32, device=device)
                for class_name in selected_classes:
                    idx = class_map[class_name]
                    combined_mask += (pred_seg == idx).float()
                
                combined_mask = torch.clamp(combined_mask, 0, 1).unsqueeze(1) # (current_B, 1, H, W)

                # Blur (GPU)
                if mask_blur > 0:
                    k_size = mask_blur * 2 + 1
                    combined_mask = gaussian_blur(combined_mask, kernel_size=k_size)

                # Offset (GPU)
                if mask_offset != 0:
                    if mask_offset > 0:
                        k_size = mask_offset * 2 + 1
                        combined_mask = F.max_pool2d(combined_mask, kernel_size=k_size, stride=1, padding=mask_offset)
                    else:
                        k_size = -mask_offset * 2 + 1
                        combined_mask = -F.max_pool2d(-combined_mask, kernel_size=k_size, stride=1, padding=-mask_offset)

                if invert_output:
                    combined_mask = 1.0 - combined_mask

                # Final composite
                mask_final = combined_mask.permute(0, 2, 3, 1) # (current_B, H, W, 1)

                if background == "Alpha":
                    res_imgs = torch.cat([batch_imgs, mask_final], dim=-1)
                else:
                    res_imgs = batch_imgs * mask_final + bg_tensor * (1.0 - mask_final)
                
                # Collect results (move to CPU immediately to free VRAM)
                result_images_list.append(res_imgs.cpu())
                mask_final_list.append(mask_final.cpu())

            # Cat results
            final_images = torch.cat(result_images_list, dim=0)
            final_masks = torch.cat(mask_final_list, dim=0)

            return (final_images, final_masks.squeeze(-1), final_masks.repeat(1, 1, 1, 3))

        except Exception as e:
            self.clear_model()
            raise RuntimeError(f"Error in Face Parsing processing: {str(e)}")
        finally:
            if self.model is not None and not self.model.training:
                self.clear_model()

NODE_CLASS_MAPPINGS = {
    "zFaceSegment": FaceSegment
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "zFaceSegment": "Face Segment (zRMBG)"
} 
