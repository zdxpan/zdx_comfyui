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
import onnxruntime
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

class BodySegment:
    def __init__(self):
        self.model = None
        self.cache_dir = os.path.join(folder_paths.models_dir, "RMBG", "body_segment")
        self.model_file = "deeplabv3p-resnet50-human.onnx"
    
    @classmethod
    def INPUT_TYPES(cls):
        available_classes = [
            "Hair", "Glasses", "Top-clothes", "Bottom-clothes", 
            "Torso-skin", "Face", "Left-arm", "Right-arm",
            "Left-leg", "Right-leg", "Left-foot", "Right-foot"
        ]
        
        tooltips = {
            "process_res": "Processing resolution (fixed at 512x512)",
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
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": tooltips["background"]}),
                "background_color": ("COLORCODE", {"default": "#222222", "tooltip": tooltips["background_color"]}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "segment_body"
    CATEGORY = "zdx/🧽RMBG"

    def check_model_cache(self):
        model_path = os.path.join(self.cache_dir, self.model_file)
        if not os.path.exists(model_path):
            return False, "Model file not found"
        return True, "Model cache verified"

    def clear_model(self):
        if self.model is not None:
            del self.model
            self.model = None

    def download_model_files(self):
        model_id = "Metal3d/deeplabv3p-resnet50-human"
        os.makedirs(self.cache_dir, exist_ok=True)
        print("Downloading body segmentation model...")
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=self.model_file,
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False
            )
            
            if os.path.dirname(downloaded_path) != self.cache_dir:
                target_path = os.path.join(self.cache_dir, self.model_file)
                shutil.move(downloaded_path, target_path)
            return True, "Model file downloaded successfully"
        except Exception as e:
            return False, f"Error downloading model file: {str(e)}"

    def segment_body(self, images, mask_blur=0, mask_offset=0, background="Alpha", background_color="#222222", invert_output=False, **class_selections):
        try:
            # Check and download model if needed
            cache_status, message = self.check_model_cache()
            if not cache_status:
                print(f"Cache check: {message}")
                download_status, download_message = self.download_model_files()
                if not download_status:
                    raise RuntimeError(download_message)
            
            # Load model if needed
            if self.model is None:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if "cuda" in device else ['CPUExecutionProvider']
                self.model = onnxruntime.InferenceSession(
                    os.path.join(self.cache_dir, self.model_file),
                    providers=providers
                )

            # Class mapping
            class_map = {
                "Hair": 2, "Glasses": 4, "Top-clothes": 5,
                "Bottom-clothes": 9, "Torso-skin": 10, "Face": 13,
                "Left-arm": 14, "Right-arm": 15, "Left-leg": 16,
                "Right-leg": 17, "Left-foot": 18, "Right-foot": 19
            }

            # Get selected classes
            selected_classes = [name for name, selected in class_selections.items() if selected]
            if not selected_classes:
                selected_classes = ["Face", "Hair", "Top-clothes", "Bottom-clothes"]

            # Batch processing setup
            B, H, W, C = images.shape
            
            # 1. Preprocess (Torch/GPU)
            # Move to device if not already (ComfyUI usually passes CPU tensors, but we want to process on GPU if possible)
            # Actually, let's keep it on the device of the input or move to CUDA if available for ops
            process_device = device
            images_device = images.to(process_device)
            
            # Resize to 512x512
            # images: (B, H, W, C) -> (B, C, H, W)
            x = images_device.permute(0, 3, 1, 2)
            if H != 512 or W != 512:
                x_small = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
            else:
                x_small = x
                
            # Normalize: [0, 1] -> [-1, 1]
            x_input = x_small * 2.0 - 1.0
            
            # Convert to numpy for ONNX
            # Model expects (B, 512, 512, 3) based on original code logic
            x_input_np = x_input.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

            # 2. Inference (ONNX)
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            
            results_list = []
            # Run inference loop
            # Optimize: Process in batches (e.g. 10) for faster inference on small models
            inference_batch_size = 10
            try:
                for i in range(0, B, inference_batch_size):
                    end_idx = min(i + inference_batch_size, B)
                    in_tensor = x_input_np[i:end_idx]
                    pred = self.model.run([output_name], {input_name: in_tensor})[0]
                    results_list.append(pred)
            except Exception as e:
                print(f"Batch inference failed, falling back to single item processing: {e}")
                results_list = []
                for i in range(B):
                    in_tensor = x_input_np[i:i+1] 
                    pred = self.model.run([output_name], {input_name: in_tensor})[0]
                    results_list.append(pred)
            
            # Stack results
            results = np.concatenate(results_list, axis=0) # (B, 512, 512, 20)
            results_tensor = torch.from_numpy(results).to(process_device)
            
            # 3. Post-process (Torch/GPU)
            # Argmax
            pred_seg = torch.argmax(results_tensor, dim=3) # (B, 512, 512)
            
            # Combine selected class masks
            combined_mask = torch.zeros((B, 512, 512), dtype=torch.float32, device=process_device)
            for class_name in selected_classes:
                idx = class_map[class_name]
                combined_mask += (pred_seg == idx).float()
            
            combined_mask = torch.clamp(combined_mask, 0, 1)
            
            # Resize mask back to original size
            combined_mask = combined_mask.unsqueeze(1) # (B, 1, 512, 512)
            if H != 512 or W != 512:
                combined_mask = F.interpolate(combined_mask, size=(H, W), mode='bilinear', align_corners=False)
            
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
                
            # Final composition
            # combined_mask is (B, 1, H, W)
            mask_final = combined_mask.permute(0, 2, 3, 1) # (B, H, W, 1)
            
            if background == "Alpha":
                # Create RGBA
                result_images = torch.cat([images_device, mask_final], dim=-1)
            else:
                # Color background
                hex_color = background_color.lstrip('#')
                if len(hex_color) == 6:
                    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                elif len(hex_color) == 8:
                    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                    # Alpha ignored for background color replacement usually? Or should we use it?
                    # Original code ignored alpha of bg color effectively or used it for composite?
                    # Original: bg_image = Image.new('RGBA', orig_image.size, rgba)
                    # Let's assume solid color for now or handle simple RGB
                else:
                    r, g, b = 0, 0, 0
                
                bg_tensor = torch.tensor([r, g, b], dtype=torch.float32, device=process_device).view(1, 1, 1, 3) / 255.0
                # Composite: image * mask + bg * (1 - mask)
                # mask is alpha for the subject
                result_images = images_device * mask_final + bg_tensor * (1.0 - mask_final)

            # Prepare outputs
            # Result images: (B, H, W, C)
            # Mask: (B, H, W)
            # Mask preview: (B, H, W, 3)
            
            return (result_images.cpu(), mask_final.squeeze(-1).cpu(), mask_final.repeat(1, 1, 1, 3).cpu())

        except Exception as e:
            # self.clear_model() # Don't clear on error immediately if we want to debug, but ok to keep safe
            raise RuntimeError(f"Error in Body Segmentation processing: {str(e)}")
        finally:
            self.clear_model()

NODE_CLASS_MAPPINGS = {
    "zBodySegment": BodySegment
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "zBodySegment": "Body Segment (zRMBG)"
} 