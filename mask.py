from PIL import Image, ImageFilter
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import copy
from typing import Union, List
import cv2

from PIL import Image, ImageDraw, ImageFont, ImageChops
from typing import Sequence, Mapping, Any, Union
from .imagecreatemask import image_concat_mask, image_concat_mask_v1, EmptyImagePro, MaskAreaReColor
from .mask_crop_v2 import MaskCropV2, CropRestore, GH_CropRestore, GH_MaskCropV2


MAX_RESOLUTION = 1024 * 16

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


class ObjExtractByMask:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "image": ("IMAGE",), "mask": ("MASK",),
                    }
                }
    CATEGORY = "zdx/mask"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "resize_obj_area_to_max_size"
### same as crop by mask 
    # def extract(self, image, masks):
    def resize_obj_area_to_max_size(self, image, mask):
        # TODO 寻找最小外接矩形，多边形      # 保持比例等比缩放算法~
        image_pil = tensor2pil(image)
        mask_pil = mask2image(mask)
            
        bbox = mask_pil.getbbox()
        if bbox is None:
            return (image,)        #  最小外接矩形, 未带~ 扩展缩放~
        cloth_obj = image_pil.crop(bbox)
        w,h = image_pil.size   #  # 原始图像的宽高
        w_,h_ = cloth_obj.size    #  # 目标区域
        scale_ = min(w/w_ , h/ h_)
        new_w,new_h = int(scale_ * w_), int(scale_ * h_)
        cloth_obj = cloth_obj.resize(size = (new_w,new_h))    # # 缩放目标区域
        # pasted back to the orignal size~
        obj_expand = Image.new(cloth_obj.mode, (w,h), (0, 0, 0))
        x = 0 if new_w == w else (w - new_w) // 2 
        y = 0 if new_h == h else (h - new_h) // 2
        obj_expand.paste(cloth_obj, (x,y))
        # obj_expand.save('1_obj_resized_inpil.jpeg')
        return (pil2tensor(obj_expand), )

class isMaskEmpty:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = "zdx/Logic"

    def execute(self, mask):
        if mask is None:
            return (True,)
        if torch.all(mask == 0):
            return (True,)
        return (False,)

class MaskSubtraction:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks_a": ("MASK",),
                        "masks_b": ("MASK",),
                    }
                }

    CATEGORY = "zdx/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)
    FUNCTION = "subtract_masks"

    def subtract_masks(self, masks_a, masks_b):
        subtracted_masks = torch.clamp(masks_a - masks_b, 0, 255)
        return (subtracted_masks,)

class MaskAdd:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks_a": ("MASK",),
                        "masks_b": ("MASK",),
                    }
                }

    CATEGORY = "zdx/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)
    FUNCTION = "subtract_masks"

    def subtract_masks(self, masks_a, masks_b):
        subtracted_masks = torch.clamp(masks_a + masks_b, 0, 255)
        return (subtracted_masks,)


class MaskCombine:

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks_a": ("MASK",),
                        "masks_b": ("MASK",),
                    }
                }

    CATEGORY = "zdx/mask"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "combine_mask"
    def combine_mask(self, masks_a, masks_b):
        # Combine selected class masks
        combined_mask = np.zeros_like(masks_b[0], dtype=np.float32)
        for mask in [masks_a, masks_a]:
            combined_mask = torch.clip(combined_mask + mask, 0, 1)
        return (combined_mask,)

class MaskSolidArea:

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "mask": ("MASK",),
                        "radius": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                        "up_ratio": ("FLOAT", {"default": 0.2, "min": 0, "max": 1, "step": 0.01}),
                        "down_ratio": ("FLOAT", {"default": 0.2, "min": 0, "max": 1, "step": 0.01}),
                        "horizontal_ratio": ("FLOAT", {"default": 0.2, "min": 0, "max": 1, "step": 0.01}),
                        "fade_ratio": ("FLOAT", {"default": 0.2, "min": 0, "max": 1, "step": 0.01}),
                        "is_full_area": ("BOOLEAN", {"default": False}),
                    }
                }

    CATEGORY = "zdx/mask"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "cut"
    def cut(self, mask, radius, up_ratio, down_ratio, horizontal_ratio, fade_ratio, is_full_area):  
        mask_pil = tensor2pil(mask)
        mask_bbox = mask_pil.getbbox()
        need_paste_mask=Image.new('L',mask_pil.size,"black")
        if mask_bbox is None:
            return (mask,)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask_bbox_w,mask_bbox_h=mask_bbox[2]-mask_bbox[0],mask_bbox[3]-mask_bbox[1]
        # expand bbox
        mask_bbox=(
            int(mask_bbox[0]-mask_bbox_w*horizontal_ratio),
            int(mask_bbox[1]-mask_bbox_h*up_ratio),
            int(mask_bbox[2]+mask_bbox_w*horizontal_ratio),
            int(mask_bbox[3]+mask_bbox_h*down_ratio),
        )
        new_bbox_w,new_bbox_h=mask_bbox[2]-mask_bbox[0],mask_bbox[3]-mask_bbox[1]
        B, H, W = mask.shape
        
        fade_h, fade_w = int(new_bbox_h * fade_ratio), int(new_bbox_w * fade_ratio)
        if is_full_area:
            fade_h, fade_w = int(H * fade_ratio), int(W * fade_ratio)

        # Initialize with ones (fully opaque) if is full area else create a mask which mask_bbox area is full 1, other area is 0
        if is_full_area:
            y_grad = torch.ones((H,), dtype=torch.float32, device=mask.device)
            x_grad = torch.ones((W,), dtype=torch.float32, device=mask.device)

            # Apply linear gradients to edges
            if fade_h > 0:
                # Top fade: 0 -> 1
                y_grad[:fade_h] = torch.linspace(0, 1, fade_h, device=mask.device)
                # Bottom fade: 1 -> 0
                y_grad[-fade_h:] = torch.linspace(1, 0, fade_h, device=mask.device)
            
            if fade_w > 0:
                # Left fade: 0 -> 1
                x_grad[:fade_w] = torch.linspace(0, 1, fade_w, device=mask.device)
                # Right fade: 1 -> 0
                x_grad[-fade_w:] = torch.linspace(1, 0, fade_w, device=mask.device)
            
            # Combine gradients (outer product) to handle corners smoothly
            grad_mask = y_grad.unsqueeze(1) * x_grad.unsqueeze(0)
            grad_mask = grad_mask.expand(B, H, W)
        else:
            grad_mask = torch.zeros((B, H, W), dtype=torch.float32, device=mask.device)
            x1, y1, x2, y2 = mask_bbox
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            
            if bbox_w > 0 and bbox_h > 0:
                y_grad = torch.ones((bbox_h,), dtype=torch.float32, device=mask.device)
                x_grad = torch.ones((bbox_w,), dtype=torch.float32, device=mask.device)
                
                if fade_h > 0:
                    fh = min(fade_h, bbox_h)
                    y_grad[:fh] = torch.linspace(0, 1, fh, device=mask.device)
                    y_grad[-fh:] = torch.linspace(1, 0, fh, device=mask.device)
                
                if fade_w > 0:
                    fw = min(fade_w, bbox_w)
                    x_grad[:fw] = torch.linspace(0, 1, fw, device=mask.device)
                    x_grad[-fw:] = torch.linspace(1, 0, fw, device=mask.device)
                
                bbox_mask = y_grad.unsqueeze(1) * x_grad.unsqueeze(0)
                
                # Intersection with image
                ix1 = max(0, x1)
                iy1 = max(0, y1)
                ix2 = min(W, x2)
                iy2 = min(H, y2)
                
                if ix2 > ix1 and iy2 > iy1:
                    grad_mask[:, iy1:iy2, ix1:ix2] = bbox_mask[iy1-y1:iy2-y1, ix1-x1:ix2-x1]

        return (grad_mask,)


class GrowMaskWithBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "incremental_expandrate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100, "step": 0.1}),
                "lerp_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decay_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }
    CATEGORY = "zdx/mask"
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "expand_mask"
    DESCRIPTION = """
# GrowMaskWithBlur
- mask: Input mask or mask batch
- expand: Expand or contract mask or mask batch by a given amount
- incremental_expandrate: increase expand rate by a given amount per frame
- tapered_corners: use tapered corners
- flip_input: flip input mask
- blur_radius: value higher than 0 will blur the mask
- lerp_alpha: alpha value for interpolation between frames
- decay_factor: decay value for interpolation between frames
- fill_holes: fill holes in the mask (slow)"""
    def expand_mask(self, mask, expand, tapered_corners, flip_input, blur_radius, incremental_expandrate, lerp_alpha, decay_factor, fill_holes=False):
        alpha = lerp_alpha
        decay = decay_factor
        if flip_input:
            mask = 1.0 - mask
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
        out = []
        previous_output = None
        current_expand = expand
        for m in growmask:
            output = m.numpy().astype(np.float32)
            for _ in range(abs(round(current_expand))):
                if current_expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            if current_expand < 0:
                current_expand -= abs(incremental_expandrate)
            else:
                current_expand += abs(incremental_expandrate)
            if fill_holes:
                binary_mask = output > 0
                output = scipy.ndimage.binary_fill_holes(binary_mask)
                output = output.astype(np.float32) * 255
            output = torch.from_numpy(output)
            if alpha < 1.0 and previous_output is not None:
                # Interpolate between the previous and current frame
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                # Add the decayed previous output to the current frame
                output += decay * previous_output
                output = output / output.max()
            previous_output = output
            out.append(output)
        if blur_radius != 0:
            # Convert the tensor list to PIL images, apply blur, and convert back
            for idx, tensor in enumerate(out):
                # Convert tensor to PIL image
                pil_image = tensor2pil(tensor.cpu().detach())
                # Apply Gaussian blur
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                # Convert back to tensor
                out[idx] = pil2tensor(pil_image)
            blurred = torch.cat(out, dim=0)
            return (blurred, 1.0 - blurred)
        else:
            return (torch.stack(out, dim=0), 1.0 - torch.stack(out, dim=0),)

class MaskExpand(GrowMaskWithBlur):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "incremental_expandrate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "lerp_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decay_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }
    CATEGORY = "zdx/mask"
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "expand_mask"
    DESCRIPTION = """
    MaskExpand no blur
    - fill_holes: fill holes in the mask (slow)"""

    def expand_mask(self, mask, expand, tapered_corners, flip_input, incremental_expandrate, lerp_alpha, decay_factor, fill_holes=False):
        return super().expand_mask(mask, expand, tapered_corners, flip_input, 0.0, incremental_expandrate, lerp_alpha, decay_factor, fill_holes)

class GrowMaskEdgeBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "incremental_expandrate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 100, "step": 0.1 }),
                "lerp_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decay_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }
    CATEGORY = "zdx/mask"
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "expand_edge_mask"
    DESCRIPTION = """
    # GrowMaskEdgeBlur
    - mask: Input mask or mask batch
    - expand: Expand or contract mask or mask batch by a given amount
    - incremental_expandrate: increase expand rate by a given amount per frame
    - tapered_corners: use tapered corners
    - flip_input: flip input mask
    - blur_radius: value higher than 0 will blur the mask
    - lerp_alpha: alpha value for interpolation between frames
    - decay_factor: decay value for interpolation between frames
    - fill_holes: fill holes in the mask (slow)"""

    def expand_edge_mask(self, mask, expand, tapered_corners, flip_input, blur_radius, incremental_expandrate, lerp_alpha, decay_factor, fill_holes=False):
        
        mask_expand_processor = MaskExpand()
        mask_sub = MaskSubtraction()
        
        out = []
        expand_out = mask_expand_processor.expand_mask(
            mask=mask, expand=expand, tapered_corners=tapered_corners, flip_input=flip_input, incremental_expandrate=incremental_expandrate, 
            lerp_alpha=lerp_alpha, decay_factor=decay_factor, fill_holes=fill_holes
        )[0]
        # to list ~
        for i in expand_out:
            out.append(i)

        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()  # original mask

        # expand_mask = copy.deepcopy(out)
        # mask2image(out[0]).save('1_2_expanded_mask.jpeg')    # checked had filed area
        # edge_mask = copy.deepcopy(out)

        if blur_radius != 0:
            # Convert the tensor list to PIL images, apply blur, and convert back
            for idx, tensor in enumerate(out):
                # Convert tensor to PIL image
                pil_image = kjtensor2pil(tensor.cpu().detach())[0]
                # Apply Gaussian blur
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                # pil_image.save('1_3_blur_mask.jpeg')
                # Convert back to tensor
                out[idx] = kjpil2tensor(pil_image)
                # edge_mask[idx] = mask_sub.subtract_masks(out[idx], growmask[idx])[0]
                edge_mask = mask_sub.subtract_masks(out[idx], growmask[idx])[0]
                # mask2image(edge_mask[idx]).save('/data/studyzdx/1_3_edge_mask.jpeg')    # checked had filed area
                combined_mask = torch.clamp(edge_mask + growmask[idx], 0, 1)
                out[idx] = combined_mask
                # mask2image(combined_mask).save('/data/studyzdx/1_4_edge_add_org_mask.jpeg')
            blurred = torch.cat(out, dim=0)
            return (blurred, 1.0 - blurred)
        else:
            return (torch.stack(out, dim=0), 1.0 - torch.stack(out, dim=0),)

# brow from kj nodes ImagePadForOutpaintMasked
class ImageMaskPad:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "zdx/image"

    def expand_image(self, image, left, top, right, bottom, feathering, mask=None):
        
        # Handle negative padding (clamping to 0) to prevent shape errors
        if left < 0 or top < 0 or right < 0 or bottom < 0:
            print(f"Warning: Negative padding detected (left={left}, top={top}, right={right}, bottom={bottom}). Clamping to 0.")
            left = max(0, left)
            top = max(0, top)
            right = max(0, right)
            bottom = max(0, bottom)

        if mask is not None:
            if torch.allclose(mask, torch.zeros_like(mask)):
                    print("Warning: The incoming mask is fully black. Handling it as None.")
                    mask = None
        
        # Ensure image is (B, H, W, C)
        if image.dim() == 3:
            image = image.unsqueeze(0)
            print(f"DEBUG: unsqueezed image shape: {image.shape}")
            
        B, H, W, C = image.size()
        
        # Robust check for BCHW vs BHWC
        # If H is very small (channels) and C is large (height/width), it's likely BCHW
        if (H <= 4) and (C > 100):
             print(f"DEBUG: Detected BCHW format {image.shape}, permuting to BHWC")
             image = image.permute(0, 2, 3, 1)
             B, H, W, C = image.size()
             print(f"DEBUG: New image shape: {image.shape}")

        new_image = torch.ones(
            (B, H + top + bottom, W + left + right, C),
            dtype=torch.float32,
        ) * 0.5

        new_image[:, top:top + H, left:left + W, :] = image

        if mask is None:
            new_mask = torch.ones(
                (B, H + top + bottom, W + left + right),
                dtype=torch.float32,
            )

            t = torch.zeros(
            (B, H, W),
            dtype=torch.float32
            )
        else:
            # If a mask is provided, pad it to fit the new image size
            mask = F.pad(mask, (left, right, top, bottom), mode='constant', value=0)
            t = torch.zeros_like(mask)
        
        if feathering > 0 and feathering * 2 < H and feathering * 2 < W:

            for i in range(H):
                for j in range(W):
                    dt = i if top != 0 else H
                    db = H - i if bottom != 0 else H

                    dl = j if left != 0 else W
                    dr = W - j if right != 0 else W

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering

                    if mask is None:
                        t[:, i, j] = v * v
                    else:
                        t[:, top + i, left + j] = v * v
        
        if mask is None:
            new_mask[:, top:top + H, left:left + W] = t
            return (new_image, new_mask,)
        else:
            return (new_image, mask,)

class ImagePadForRemove:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE","MASK", "PADINFO")
    RETURN_NAMES = ("image","mask","padinfo")
    FUNCTION = "expand_image"
    CATEGORY = "zdx/image"

    def expand_image(self, image, mask=None, ratio=1.0):
        mask_pil = mask2image(mask)
        bbox = mask_pil.getbbox()
        w,h = image.shape[2], image.shape[1]
        if bbox is None:
            return (image, mask, None)
        x, y, x_w, y_h = bbox
        w_, h_ = x_w - x, y_h - y
        old_ratio_ = min(w/w_ , h/ h_)
        if old_ratio_ >= ratio:
            return (image, mask, None)
        new_w, new_h = int(ratio * w_), int(ratio * h_)
        # need pad 
        pad_left = (new_w - w) // 2
        pad_top = (new_h - h) // 2
        pad_right = new_w - w - pad_left
        pad_bottom = new_h - h - pad_top

        pad_left = max(0, pad_left)
        pad_right = max(0, pad_right)
        pad_top = max(0, pad_top)
        pad_bottom = max(0, pad_bottom)

        padinfo = (pad_left, pad_top, pad_right, pad_bottom)
        # pad image and mask
        image_mask_pad_1 = ImageMaskPad().expand_image(
            image=image,
            mask=mask,
            left=pad_left, top=pad_top, right=pad_right, bottom=pad_bottom, feathering=0, 
        )
        print(f"DEBUG: padinfo: {padinfo}", image_mask_pad_1[0].shape, image_mask_pad_1[1].shape)
        return (image_mask_pad_1[0], image_mask_pad_1[1], padinfo)        

class ImageMaskUnPad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padinfo": ("PADINFO",),
            },
        }

    RETURN_TYPES = ("IMAGE","MASK")
    RETURN_NAMES = ("image","mask")
    FUNCTION = "expand_image"
    CATEGORY = "zdx/image"

    def expand_image(self, image, mask=None, padinfo=None):
        # 根据padinfo信息crop 还原image和mask， 如果padinfo 四个元素为0 ，则直接返回原图和mask
        if padinfo is None or all(x == 0 for x in padinfo):
            return (image, mask)
        # Check if pad is new 4-element format or old 2-element format
        pad_left, pad_top, pad_right, pad_bottom = padinfo
        print(f"DEBUG: padinfo pad_left, pad_top, pad_right, pad_bottom: {padinfo}")
        # Use specific padding values for each side
        end_h = -pad_bottom if pad_bottom > 0 else None
        end_w = -pad_right if pad_right > 0 else None
        
        # Start index is simply top/left padding
        start_h = pad_top
        start_w = pad_left
        
        croped_image_unpad = image[:, start_h:end_h, start_w:end_w, :]
        croped_mask_unpad = mask
        if mask is not None:
            croped_mask_unpad = mask[:, start_h:end_h, start_w:end_w]

        return (croped_image_unpad, croped_mask_unpad)

# image_concat_mask, image_concat_mask_v1, EmptyImagePro, MaskAreaReColor
_NODE_CLASS_MAPPINGS = {
    "GrowMaskEdgeBlur": GrowMaskEdgeBlur,
    "GrowMaskWithBlur": GrowMaskWithBlur,
    "MaskExpand": MaskExpand,
    "MaskCombine": MaskCombine,
    "MaskAdd": MaskAdd,
    "MaskSolidArea": MaskSolidArea,
    "MaskSubtraction": MaskSubtraction,
    "isMaskEmpty": isMaskEmpty,
    "EmptyImagePro": EmptyImagePro,
    "image_concat_mask": image_concat_mask,
    "image_concat_mask_v1": image_concat_mask_v1,
    "MaskAreaReColor": MaskAreaReColor,

    "MaskCropV2": MaskCropV2,
    "CropRestore": CropRestore,
    "GH_MaskCropV2": GH_MaskCropV2,
    "GH_CropRestore": GH_CropRestore,
    "ImageMaskPad": ImageMaskPad,
    "ImagePadForRemove": ImagePadForRemove,
    "ImageMaskUnPad": ImageMaskUnPad,
    

}
_NODE_DISPLAY_NAME_MAPPINGS = {
    "GrowMaskEdgeBlur": "GrowMaskEdgeBlur", 
    "zdxGrowMaskWithBlur": "GrowMaskWithBlur",
    "MaskExpand": "MaskExpand",
    "MaskCombine": "MaskCombine",
    "MaskAdd": "MaskAdd",
    "MaskSolidArea": "MaskSolidArea",
    "MaskSubtraction": "MaskSubtraction",
    "isMaskEmpty": "isMaskEmpty",
    "EmptyImagePro": "EmptyImagePro",
    "image_concat_mask": "image_concat_mask",
    "image_concat_mask_v1": "image_concat_mask_v1",
    "MaskAreaReColor": "MaskAreaReColor",

    "MaskCropV2": "Mask Crop V2",
    "CropRestore": "Crop Restore",
    "GH_MaskCropV2" :"GH_MaskCropV2",
    "GH_CropRestore": "GH_CropRestore",
    "ImageMaskPad": "ImageMaskPad",
    "ImagePadForRemove": "ImagePadForRemove",
    "ImageMaskUnPad": "ImageMaskUnPad",
}
