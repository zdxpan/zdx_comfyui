
'''
尤其是在 Merged 模式下处理视频时：

1. 冗余的 Mask 生成与图像合成 ：原代码的 _run_single_merged 方法会调用 _run_single_per_instance ，后者会为 每一个 检测到的片段（segment）生成单独的合成图像（composited image）。然后， _run_single_merged 却只使用了 Mask 张量，完全丢弃了那些耗时的合成图像，并重新进行一次合成。对于包含多个对象的视频帧，这造成了极大的计算浪费。
2. 冗余的图像转换 ：在 Merged 模式下， tensor2pil 被调用了两次（一次在外部，一次在内部）。
优化方案 ：
我已对 AILab_SAM3Segment.py 进行了重构，删除了低效的辅助函数，并直接在 segment 主循环中实现了优化的逻辑：

- 提取核心预测逻辑 ：新建 _get_masks 方法，仅负责模型推理和 Mask 筛选，不进行任何图像合成。
- 优化 Merged 模式 ：直接使用预测出的 Raw Masks 进行合并（ amax ），然后只进行 一次 最终图像的合成处理。
- 优化 Separate 模式 ：保持原有的逐 Mask 处理逻辑，但消除了冗余的类型转换。
预期效果 ：
对于视频处理（批量输入），在默认的 Merged 模式下，将显著减少 CPU 端的图像处理开销（PIL 操作），大幅提升处理速度。
'''
import os
import sys
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.hub import download_url_to_file

import folder_paths
import comfy.model_management

from AILab_ImageMaskTools import pil2tensor, tensor2pil

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SAM3_LOCAL_DIR = REPO_ROOT / "models" / "sam3"
if str(SAM3_LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(SAM3_LOCAL_DIR))
MODELS_ROOT = REPO_ROOT / "models"
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

SAM3_BPE_PATH = SAM3_LOCAL_DIR / "assets" / "bpe_simple_vocab_16e6.txt.gz"
if not os.path.isfile(SAM3_BPE_PATH):
    raise RuntimeError("SAM3 assets missing; ensure sam3/assets/bpe_simple_vocab_16e6.txt.gz exists.")

_DEFAULT_PT_ENTRY = {
    "model_url": "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt",
    "filename": "sam3.pt",
}

SAM3_MODELS = {
    "sam3": _DEFAULT_PT_ENTRY.copy(),
}


def get_sam3_pt_models():
    entry = SAM3_MODELS.get("sam3")
    if entry and entry.get("filename", "").endswith(".pt"):
        return {"sam3": entry}
    for key, value in SAM3_MODELS.items():
        if value.get("filename", "").endswith(".pt"):
            return {"sam3": value}
        if "sam3" in key and value:
            candidate = value.copy()
            candidate["model_url"] = _DEFAULT_PT_ENTRY["model_url"]
            candidate["filename"] = _DEFAULT_PT_ENTRY["filename"]
            return {"sam3": candidate}
    return {"sam3": _DEFAULT_PT_ENTRY.copy()}


def process_mask(mask_image, invert_output=False, mask_blur=0, mask_offset=0):
    if invert_output:
        mask_np = np.array(mask_image)
        mask_image = Image.fromarray(255 - mask_np)
    if mask_blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))
    if mask_offset != 0:
        filt = ImageFilter.MaxFilter if mask_offset > 0 else ImageFilter.MinFilter
        size = abs(mask_offset) * 2 + 1
        for _ in range(abs(mask_offset)):
            mask_image = mask_image.filter(filt(size))
    return mask_image


def apply_background_color(image, mask_image, background="Alpha", background_color="#222222"):
    rgba_image = image.copy().convert("RGBA")
    rgba_image.putalpha(mask_image.convert("L"))
    if background == "Color":
        hex_color = background_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        bg_image = Image.new("RGBA", image.size, (r, g, b, 255))
        composite = Image.alpha_composite(bg_image, rgba_image)
        return composite.convert("RGB")
    return rgba_image


def get_or_download_model_file(filename, url):
    local_path = None
    if hasattr(folder_paths, "get_full_path"):
        local_path = folder_paths.get_full_path("sam3", filename)
    if local_path and os.path.isfile(local_path):
        return local_path
    base_models_dir = getattr(folder_paths, "models_dir", os.path.join(CURRENT_DIR, "models"))
    models_dir = os.path.join(base_models_dir, "sam3")
    os.makedirs(models_dir, exist_ok=True)
    local_path = os.path.join(models_dir, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from {url} ...")
        download_url_to_file(url, local_path)
    return local_path


def _resolve_device(user_choice):
    auto_device = comfy.model_management.get_torch_device()
    if user_choice == "CPU":
        return torch.device("cpu")
    if user_choice == "GPU":
        if auto_device.type != "cuda":
            raise RuntimeError("GPU unavailable")
        return torch.device("cuda")
    return auto_device


from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402


class SAM3Segment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Describe the concept"}),
                "output_mode": (["Merged", "Separate"], {"default": "Merged"}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.01}),
            },
            "optional": {
                "max_segments": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "segment_pick": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                "device": (["Auto", "CPU", "GPU"], {"default": "Auto"}),
                "invert_output": ("BOOLEAN", {"default": False}),
                "unload_model": ("BOOLEAN", {"default": False}),
                "background": (["Alpha", "Color"], {"default": "Alpha"}),
                "background_color": ("COLORCODE", {"default": "#222222"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "segment"
    CATEGORY = "zdx/🧽RMBG"

    def __init__(self):
        self.processor_cache = {}

    def _load_processor(self, device_choice):
        torch_device = _resolve_device(device_choice)
        device_str = "cuda" if torch_device.type == "cuda" else "cpu"
        cache_key = ("sam3", device_str)
        if cache_key not in self.processor_cache:
            model_info = SAM3_MODELS["sam3"]
            ckpt_path = get_or_download_model_file(model_info["filename"], model_info["model_url"])
            model = build_sam3_image_model(
                bpe_path=SAM3_BPE_PATH,
                device=device_str,
                eval_mode=True,
                checkpoint_path=ckpt_path,
                load_from_HF=False,
                enable_segmentation=True,
                enable_inst_interactivity=False,
            )
            processor = Sam3Processor(model, device=device_str)
            self.processor_cache[cache_key] = processor
        return self.processor_cache[cache_key], torch_device

    def _empty_result(self, img_pil, background, background_color):
        w, h = img_pil.size
        mask_image = Image.new("L", (w, h), 0)
        result_image = apply_background_color(img_pil, mask_image, background, background_color)
        result_image = result_image.convert("RGBA") if background == "Alpha" else result_image.convert("RGB")
        empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
        mask_rgb = empty_mask.reshape((-1, 1, h, w)).movedim(1, -1).expand(-1, -1, -1, 3)
        return result_image, empty_mask, mask_rgb

    def _empty_batch(self, img_pil):
        w, h = img_pil.size
        empty_imgs = torch.zeros((0, h, w, 3), dtype=torch.float32)
        empty_masks = torch.zeros((0, h, w), dtype=torch.float32)
        empty_mask_images = torch.zeros((0, h, w, 3), dtype=torch.float32)
        return empty_imgs, empty_masks, empty_mask_images

    def _get_masks(self, processor, img_pil, prompt, confidence, max_segments, segment_pick):
        text = prompt.strip() or "object"
        state = processor.set_image(img_pil)
        processor.reset_all_prompts(state)
        processor.set_confidence_threshold(confidence, state)
        state = processor.set_text_prompt(text, state)
        masks = state.get("masks")
        logits = state.get("masks_logits")

        if masks is None or masks.numel() == 0:
            return None

        masks = masks.float()
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        scores = None
        if logits is not None:
            logits = logits.float()
            if logits.ndim == 4:
                logits = logits.squeeze(1)
            scores = logits.mean(dim=(-2, -1))

        if scores is None:
            scores = torch.ones((masks.shape[0],), device=masks.device)

        if max_segments > 0 and masks.shape[0] > max_segments:
            topk = torch.topk(scores, k=max_segments)
            masks = masks[topk.indices]
            scores = scores[topk.indices]

        sorted_idx = torch.argsort(scores, descending=True)
        masks = masks[sorted_idx]

        if segment_pick > 0:
            idx = segment_pick - 1
            if idx >= masks.shape[0]:
                return None
            masks = masks[idx : idx + 1]

        return masks

    def segment(self, image, prompt, device, confidence_threshold=0.5, max_segments=0, segment_pick=0, mask_blur=0, mask_offset=0, invert_output=False, unload_model=False, background="Alpha", background_color="#222222", output_mode="Merged"):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        processor, torch_device = self._load_processor(device)
        autocast_device = comfy.model_management.get_autocast_device(torch_device)
        autocast_enabled = torch_device.type == "cuda" and not comfy.model_management.is_device_mps(torch_device)
        ctx = torch.autocast(autocast_device, dtype=torch.bfloat16) if autocast_enabled else nullcontext()
        result_images, result_masks, result_mask_images = [], [], []
        with ctx:
            for tensor_img in image:
                img_pil = tensor2pil(tensor_img)
                masks = self._get_masks(processor, img_pil, prompt, confidence_threshold, max_segments, segment_pick)

                if masks is None or masks.shape[0] == 0:
                    if output_mode == "Separate":
                        e_imgs, e_masks, e_m_imgs = self._empty_batch(img_pil)
                        result_images.append(e_imgs)
                        result_masks.append(e_masks)
                        result_mask_images.append(e_m_imgs)
                    else:
                        r_img, r_mask, r_m_img = self._empty_result(img_pil, background, background_color)
                        result_images.append(pil2tensor(r_img))
                        result_masks.append(r_mask)
                        result_mask_images.append(r_m_img)
                    continue

                if output_mode == "Separate":
                    mask_imgs, mask_tensors, mask_rgb_list = [], [], []
                    for single_mask in masks:
                        mask_np = (single_mask.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                        mask_image = Image.fromarray(mask_np, mode="L")
                        mask_image = process_mask(mask_image, invert_output, mask_blur, mask_offset)
                        composed = apply_background_color(img_pil, mask_image, background, background_color)
                        composed = composed.convert("RGBA") if background == "Alpha" else composed.convert("RGB")
                        mask_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
                        mask_rgb = mask_tensor.reshape((1, mask_image.height, mask_image.width, 1)).expand(-1, -1, -1, 3)
                        mask_imgs.append(pil2tensor(composed))
                        mask_tensors.append(mask_tensor)
                        mask_rgb_list.append(mask_rgb)
                    result_images.append(torch.cat(mask_imgs, dim=0))
                    result_masks.append(torch.cat(mask_tensors, dim=0))
                    result_mask_images.append(torch.cat(mask_rgb_list, dim=0))
                else:
                    merged = masks.amax(dim=0)
                    mask_np = (merged.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    mask_image = Image.fromarray(mask_np, mode="L")
                    mask_image = process_mask(mask_image, invert_output, mask_blur, mask_offset)
                    result_image = apply_background_color(img_pil, mask_image, background, background_color)
                    result_image = result_image.convert("RGBA") if background == "Alpha" else result_image.convert("RGB")
                    mask_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
                    mask_rgb = mask_tensor.reshape((1, mask_image.height, mask_image.width, 1)).expand(-1, -1, -1, 3)
                    result_images.append(pil2tensor(result_image))
                    result_masks.append(mask_tensor)
                    result_mask_images.append(mask_rgb)

        if unload_model:
            device_str = "cuda" if torch_device.type == "cuda" else "cpu"
            cache_key = ("sam3", device_str)
            if cache_key in self.processor_cache:
                del self.processor_cache[cache_key]
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

        final_images = torch.cat(result_images, dim=0)
        final_masks = torch.cat(result_masks, dim=0)
        final_mask_images = torch.cat(result_mask_images, dim=0)
        if final_images.shape[0] == 0:
            img_pil = tensor2pil(image[0])
            empty_img, empty_mask, empty_mask_img = self._empty_result(img_pil, background, background_color)
            final_images = pil2tensor(empty_img)
            final_masks = empty_mask
            final_mask_images = empty_mask_img
        return final_images, final_masks, final_mask_images


try:
    from ultralytics import SAM
    from ultralytics.models.sam import SAM3SemanticPredictor
except ImportError:
    SAM = None
    SAM3SemanticPredictor = None


class UltralyticsSAM3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (["sam3.pt", "sam3_tiny.pt", "sam3_small.pt", "sam3_base.pt", "sam3_large.pt"], {"default": "sam3.pt"}),
            },
            "optional": {
                "text_prompt": ("STRING", {"multiline": True, "placeholder": "Enter text prompts (e.g., 'person', 'car'). Split by line or comma."}),
                "bbox_prompt": ("STRING", {"multiline": True, "placeholder": "x1,y1,x2,y2 (one box per line)"}),
                "point_prompt": ("STRING", {"multiline": True, "placeholder": "x,y,label (label: 1=fg, 0=bg). One point per line."}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "device": (["Auto", "CPU", "GPU"], {"default": "Auto"}),
                "force_cpu": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "segment"
    CATEGORY = "zdx/🧽RMBG"

    def segment(self, image, model_name, text_prompt, bbox_prompt, point_prompt, confidence_threshold, device, force_cpu):
        if SAM is None:
            raise ImportError("Ultralytics package is not installed. Please install it with 'pip install ultralytics'.")

        # Resolve device
        if force_cpu:
            device_str = "cpu"
        elif device == "Auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "GPU":
            device_str = "cuda"
        else:
            device_str = "cpu"

        # Load Model
        try:
            # Ultralytics auto-downloads if model is a filename and not found
            model = SAM(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise e

        # Prepare prompts
        text_prompts = [p.strip() for p in text_prompt.splitlines() if p.strip()] if text_prompt else []
        if not text_prompts and text_prompt:
             text_prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]

        bboxes = []
        if bbox_prompt:
            for line in bbox_prompt.splitlines():
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    try:
                        bboxes.append([float(p) for p in parts[:4]])
                    except ValueError:
                        pass
        
        points = []
        point_labels = []
        if point_prompt:
            for line in point_prompt.splitlines():
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        label = int(parts[2]) if len(parts) > 2 else 1
                        points.append([x, y])
                        point_labels.append(label)
                    except ValueError:
                        pass

        results_images = []
        results_masks = []
        results_mask_images = []

        # Process batch
        for i in range(image.shape[0]):
            img_tensor = image[i]
            img_pil = tensor2pil(img_tensor)
            
            masks = []
            
            # 1. Text Prompts
            if text_prompts:
                try:
                    overrides = dict(conf=confidence_threshold, task="segment", mode="predict", model=model_name, device=device_str)
                    predictor = SAM3SemanticPredictor(overrides=overrides)
                    # predictor.set_image expects numpy array (H, W, C)
                    img_np = np.array(img_pil)
                    predictor.set_image(img_np)
                    
                    # predictor call returns list of Results
                    res = predictor(text=text_prompts)
                    for r in res:
                        if r.masks is not None:
                             masks.append(r.masks.data) # (N, H, W)
                except Exception as e:
                    print(f"Error with text prompts: {e}")

            # 2. Visual Prompts (BBox / Points) / Auto
            # If we have visual prompts OR no text prompts, we might want to run standard SAM inference
            # But if we have text prompts, maybe we don't want to run auto unless requested?
            # Assuming if visual prompts provided, run them.
            # If NO prompts at all, run auto.
            
            run_visual = bool(bboxes or points)
            run_auto = not text_prompts and not run_visual
            
            if run_visual or run_auto:
                kwargs = {"conf": confidence_threshold, "device": device_str}
                if bboxes:
                    kwargs["bboxes"] = bboxes
                if points:
                    kwargs["points"] = points
                    kwargs["labels"] = point_labels
                
                # SAM model() call
                res = model(img_pil, **kwargs)
                for r in res:
                    if r.masks is not None:
                        masks.append(r.masks.data)

            # Process masks
            if not masks:
                 h, w = img_pil.size[1], img_pil.size[0]
                 empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
                 results_masks.append(empty_mask)
                 results_images.append(img_tensor.unsqueeze(0))
                 results_mask_images.append(torch.zeros((1, h, w, 3), dtype=torch.float32))
                 continue

            # Concatenate all masks found for this image (N, H, W)
            all_masks = torch.cat(masks, dim=0) 
            
            # Merge to single mask per image (H, W)
            merged_mask = all_masks.any(dim=0).float()
            results_masks.append(merged_mask.unsqueeze(0))
            
            # Apply background
            mask_pil = Image.fromarray((merged_mask.cpu().numpy() * 255).astype(np.uint8), mode="L")
            composed = apply_background_color(img_pil, mask_pil, background="Alpha")
            results_images.append(pil2tensor(composed))
            
            # Mask Image
            m = merged_mask.unsqueeze(0)
            mask_rgb = m.reshape((1, m.shape[1], m.shape[2], 1)).expand(-1, -1, -1, 3)
            results_mask_images.append(mask_rgb)

        return (torch.cat(results_images, dim=0), torch.cat(results_masks, dim=0), torch.cat(results_mask_images, dim=0))


NODE_CLASS_MAPPINGS = {
    "zSAM3Segment": SAM3Segment,
    "zUltralyticsSAM3": UltralyticsSAM3
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "zSAM3Segment": "SAM3 Segmentation (zRMBG)",
    "zUltralyticsSAM3": "Ultralytics SAM3 (zRMBG)"
}


