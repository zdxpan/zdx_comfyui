# must run at custom_nodes after custom nodes loaded!
import torch
from typing import Sequence, Mapping, Any, Union
from nodes import NODE_CLASS_MAPPINGS
from .mask import MaskAdd, MaskSubtraction, MaskExpand, GrowMaskEdgeBlur, GrowMaskWithBlur, MaskCombine, ImageMaskPad
from PIL import Image
import numpy as np
from .layerstyle import (
    ImageScaleByAspectRatioV2,
    LS_ImageMaskScaleAsV2, ImageMaskScaleAs,
    ImageScaleRestore, ImageScaleRestoreV2,
    CropByMask,  CropByMaskV2, 
    CropBoxResolve,
    RestoreCropBox,
    ImageBlendAdvanceV2, ColorImageV2
)
MAX_RESOLUTION = 8096
def pil2tensor(image):
    new_image = image.convert('RGB')
    img_array = np.array(new_image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)[None]
    return img_tensor

def tensor2pil(image):
    if len(image.shape) < 3:
        image = image.unsqueeze(0)
    return Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

# - FocusCrop 优化：实现 Batch BBox 计算 :
# - 使用 torch.max(mask, dim=0)[0] 计算整个视频序列的并集 Mask。
# - 基于这个并集 Mask 计算 BBox，确保剪裁框能覆盖所有帧中的主体运动范围。
# - 使用 torch.nonzero() 替代 PIL.getbbox() 。
# - 完全在 Tensor 层面计算坐标和尺寸。
# - 确保下游调用兼容 Batch :
# - 检查并优化 ImageMaskPad 或直接在当前节点实现 Tensor Padding。
class FocusCrop():
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "image": ("IMAGE",),
                        "up_keep": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 3, "step": 0.05, }),
                        "down_keep": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 3, "step": 0.05, }),
                        "right_keep": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 3, "step": 0.05, }),
                        "left_keep": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 3, "step": 0.05, }),
                        "is_focus": ("BOOLEAN", {"default": True}),
                        "crop_ratio": (["original", "1:1", "4:3", "3:4", "16:9", "9:16", "custom"], {"default": "original"}),
                    },
                    "optional": {
                        "all_mask": ("MASK",),
                        "mask": ("MASK",),
                    }
                }
    CATEGORY = "zdx/image"
    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "IMAGE", "MASK", "INT", "INT", "INFO")
    RETURN_NAMES = ("croped_image", "croped_mask", "crop_box", "box_preview", "croped_all_mask", "crop_h", "crop_w", "info")
    FUNCTION = 'crop_by_mask_v2'
    DESCRIPTION = """
focuscrop  crop_by_mask_v2
"""
    @torch.inference_mode()     
    def crop_by_mask_v2(self, image, all_mask=None, up_keep=0.2, down_keep=0.2, right_keep=0.2, left_keep=0.2,  is_focus=True, mask=None, crop_ratio="original"):
        self.zdxmaskadd = MaskAdd()
        self.layerutility_cropbymask_v2 = CropByMaskV2()
        self.layerutility_restorecropbox = RestoreCropBox()
        self.layerutility_imagescalebyaspectratio_v2 = ImageScaleByAspectRatioV2()
        self.layerutility_imagescalerestore_v2 = ImageScaleRestore()
        self.layerutility_cropboxresolve = CropBoxResolve()
        
        ratio_map = {"1:1": 1.0, "4:3": 4/3, "3:4": 3/4, "16:9": 16/9, "9:16": 9/16,}

        B, H, W, C = image.shape
        _, crop_h, crop_w, _ = image.shape
        
        # 1. Early Return
        if not is_focus or mask is None:
            return (image, all_mask, None, None, all_mask, crop_h, crop_w, None)

        # 2. Mask Handling (Batch Compatible)
        if all_mask is not None:
             if all_mask.dim() == 2: all_mask = all_mask.unsqueeze(0)
        
        if mask.dim() == 2: mask = mask.unsqueeze(0)
        
        NEED_CROP_TWICE = False
        if all_mask is None:
            all_mask = mask
        else:
            NEED_CROP_TWICE = True
            # Tensor subtraction: mask - all_mask
            # Broadcast
            if mask.shape[0] != all_mask.shape[0]:
                max_b = max(mask.shape[0], all_mask.shape[0])
                if mask.shape[0] == 1: mask = mask.repeat(max_b, 1, 1)
                if all_mask.shape[0] == 1: all_mask = all_mask.repeat(max_b, 1, 1)
            
            all_mask = torch.clamp(mask - all_mask, 0.0, 1.0)

        # 3. Calculate Union BBox (Tensor Op) - Supports Video Batch
        combined_mask = torch.max(all_mask, dim=0)[0] # [H, W]
        nonzero = torch.nonzero(combined_mask > 0.05)

        if nonzero.numel() == 0:
            x, y, x_w, y_h = 0, 0, W, H
        else:
            y = nonzero[:, 0].min().item()
            y_h = nonzero[:, 0].max().item() + 1 # +1 for exclusive bound
            x = nonzero[:, 1].min().item()
            x_w = nonzero[:, 1].max().item() + 1
            
        crop_w = x_w - x
        crop_h = y_h - y
        
        # 4. Calculate Reserves
        top_pix = int(crop_h * up_keep)
        down_pix = int(crop_h * down_keep)
        right_pix = int(crop_h * right_keep)
        left_pix = int(crop_h * left_keep)
        
        # 5. Ratio Adjustment
        total_w = crop_w + left_pix + right_pix
        total_h = crop_h + top_pix + down_pix
        total_w = total_w - total_w % 8
        total_h = total_h - total_h % 8
        
        if crop_ratio != "original":
            target_ratio = ratio_map.get(crop_ratio)
            if target_ratio:
                curr_ratio = total_w / total_h
                
                if curr_ratio > target_ratio:
                    # Too wide -> Increase Height
                    needed_h = int(total_w / target_ratio)
                    diff_h = needed_h - total_h
                    total_h = needed_h
                    
                    half_diff = diff_h // 2
                    top_pix += half_diff + (diff_h % 2)
                    down_pix += half_diff
                else:
                    # Too tall -> Increase Width
                    needed_w = int(total_h * target_ratio)
                    diff_w = needed_w - total_w
                    total_w = needed_w
                    
                    half_diff = diff_w // 2
                    left_pix += half_diff + (diff_w % 2)
                    right_pix += half_diff

                # Shift Logic (Try to keep in bounds)
                # Left
                if x - left_pix < 0:
                    shift = -(x - left_pix)
                    left_pix -= shift
                    right_pix += shift
                # Right
                if (x + crop_w) + right_pix > W:
                    shift = ((x + crop_w) + right_pix) - W
                    right_pix -= shift
                    left_pix += shift
                # Top
                if y - top_pix < 0:
                    shift = -(y - top_pix)
                    top_pix -= shift
                    down_pix += shift
                # Bottom
                if (y + crop_h) + down_pix > H:
                    shift = ((y + crop_h) + down_pix) - H
                    down_pix -= shift
                    top_pix += shift

        # 6. Calculate Final Crop Box (including potential out-of-bounds for padding)
        final_x1 = x - left_pix
        final_y1 = y - top_pix
        final_x2 = x + crop_w + right_pix
        final_y2 = y + crop_h + down_pix
        
        # CropByMaskV2 handles the padding automatically if we pass this box
        crop_box = (final_x1, final_y1, final_x2, final_y2)
        
        # 7. Execute Crop
        image_mask_cropbymask_1 = self.layerutility_cropbymask_v2.crop_by_mask_v2(
            invert_mask=False, detect="mask_area", top_reserve=0, bottom_reserve=0,
            left_reserve=0, right_reserve=0, round_to_multiple="None",
            image=image, mask=all_mask, crop_box=crop_box
        )
        
        final_img = image_mask_cropbymask_1[0]
        final_mask = image_mask_cropbymask_1[1]
        final_box = image_mask_cropbymask_1[2]
        
        # 8. Second Crop (if needed to return the original mask cropped)
        if NEED_CROP_TWICE:
             res_2 = self.layerutility_cropbymask_v2.crop_by_mask_v2(
                invert_mask=False, detect="mask_area",
                top_reserve=0, bottom_reserve=0, left_reserve=0, right_reserve=0, round_to_multiple="None",
                image=image, mask=mask, crop_box=crop_box
            )
             final_mask = res_2[1] # Use the cropped 'mask'
        
        # 9. Info Construction
        pl = max(0, -final_x1)
        pt = max(0, -final_y1)
        pr = max(0, final_x2 - W)
        pb = max(0, final_y2 - H)
        
        info = {"bbox": [x, y, x+crop_w, y+crop_h], 
                "pad": [pl, pt, pr, pb], 
                "crop_box": final_box}
                
        return (final_img, final_mask, final_box, image_mask_cropbymask_1[3], None, final_img.shape[1], final_img.shape[2], info)

class FocusCropV2(FocusCrop):

    @classmethod
    def INPUT_TYPES(cls):
        ratio_list = ['original', 'custom', '1:1', '3:2', '4:3', '16:9', '2:3', '3:4', '9:16']
        fit_mode = ['fill', 'letterbox', 'crop',]
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        scale_to_list = ['None', 'longest', 'shortest', 'width', 'height', 'total_pixel(kilo pixel)']
        return {
                    "required": {
                        "image": ("IMAGE",),
                        "up_keep": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 3, "step": 0.05, }),
                        "down_keep": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 3, "step": 0.05, }),
                        "right_keep": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 3, "step": 0.05, }),
                        "left_keep": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 3, "step": 0.05, }),
                        "is_focus": ("BOOLEAN", {"default": True}),
                        "aspect_ratio": (ratio_list,),
                        "proportional_width": ("INT", {"default": 1, "min": 1, "max": 1e8, "step": 1}),
                        "proportional_height": ("INT", {"default": 1, "min": 1, "max": 1e8, "step": 1}),
                        "fit": (fit_mode,),
                        "method": (method_mode,),
                        "round_to_multiple": (multiple_list,),
                        "scale_to_side": (scale_to_list,),  # 是否按长边缩放
                        "scale_to_length": ("INT", {"default": 1024, "min": 4, "max": 1e8, "step": 1}),
                        "background_color": ("STRING", {"default": "#000000"}),  # 背景颜色
                    },
                    "optional": {
                        "all_mask": ("MASK",),
                        "mask": ("MASK",),
                    }
                }
    CATEGORY = "zdx/image"
    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "IMAGE", "MASK", "INT", "INT", "BOX",
                    "IMAGE", "MASK", "BOX", "INT", "INT",)
    RETURN_NAMES = ("croped_image", "croped_mask", "crop_box", "box_preview", "croped_all_mask", "crop_h", "crop_w", "bbox", 
                    "scale_image", "scale_mask", "original_size", "width", "height",  )
    FUNCTION = 'crop_by_mask_v2'
    DESCRIPTION = """
focuscrop V2 crop_by_mask_v2
"""
    @torch.inference_mode()     
    def crop_by_mask_v2(self, image, all_mask=None, up_keep=0.2, down_keep=0.2, right_keep=0.2, left_keep=0.2,  is_focus=True, mask=None, 
            aspect_ratio='original', proportional_width=1, proportional_height=1,
            fit='fill', method='lanczos', round_to_multiple=8, scale_to_side='longest', scale_to_length=1024, background_color='#000000',
        ):
        self.focus_crop = FocusCrop()
        self.layerutility_imagescalebyaspectratio_v2 = ImageScaleByAspectRatioV2()

        if not is_focus or mask is None:
            _, crop_h, crop_w, _ = image.shape
            return (image, all_mask, None, None, all_mask, crop_h, crop_w, )

        image_mask_cropbymask_1 =  self.focus_crop.crop_by_mask_v2(
            # keep_ratio=init_crop_keep_ratio,
            up_keep=up_keep, down_keep=down_keep, right_keep=right_keep, left_keep=left_keep,
            image=image,
            all_mask=all_mask,
            mask=mask,
        )

        layerutility_imagescalebyaspectratio_v2_148 = (
            self.layerutility_imagescalebyaspectratio_v2.image_scale_by_aspect_ratio(
                aspect_ratio=aspect_ratio,
                proportional_width=proportional_width,
                proportional_height=proportional_height,
                fit=fit,
                method=method,
                round_to_multiple=round_to_multiple,
                scale_to_side=scale_to_side,
                scale_to_length=scale_to_length,
                background_color=background_color,
                image=image_mask_cropbymask_1[0],
                mask=image_mask_cropbymask_1[1],
            )
        )

        return (
            image_mask_cropbymask_1[0], 
            image_mask_cropbymask_1[1], 
            image_mask_cropbymask_1[2], 
            image_mask_cropbymask_1[3],
            image_mask_cropbymask_1[4], 
            image_mask_cropbymask_1[5], image_mask_cropbymask_1[6], # crop_h, crop_w,
            image_mask_cropbymask_1[7], # bbox,
            layerutility_imagescalebyaspectratio_v2_148[0], layerutility_imagescalebyaspectratio_v2_148[1], 
            layerutility_imagescalebyaspectratio_v2_148[2], layerutility_imagescalebyaspectratio_v2_148[3], layerutility_imagescalebyaspectratio_v2_148[4],
        )

class FocusCropUltra(FocusCrop):
    @classmethod
    def INPUT_TYPES(cls):
        ratio_list = ['original', 'custom', '1:1', '3:2', '4:3', '16:9', '2:3', '3:4', '9:16']
        fit_mode = ['fill', 'letterbox', 'crop',]
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        scale_to_list = ['None', 'longest', 'shortest', 'width', 'height', 'total_pixel(kilo pixel)']
        return {
                    "required": {
                        "image": ("IMAGE",),
                        "up_keep": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 3, "step": 0.05, }),
                        "down_keep": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 3, "step": 0.05, }),
                        "right_keep": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 3, "step": 0.05, }),
                        "left_keep": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 3, "step": 0.05, }),
                        # "is_focus": ("BOOLEAN", {"default": True}),
                        "crop_ratio": (["original", "1:1"], {"default": "original"}),
                        "aspect_ratio": (ratio_list, {"default": 'original'}),
                        "fit": (fit_mode, {"default": 'fill'}),
                        "method": (method_mode, {"default": 'lanczos'}),
                        "round_to_multiple": (multiple_list, {"default": '8'}),
                        "scale_to_side": (scale_to_list, {"default": 'longest'}),  # 是否按长边缩放
                        "scale_to_length": ("INT", {"default": 1024, "min": 4, "max": 1e8, "step": 1}),
                        "background_color": ("STRING", {"default": "#000000"}),  # 背景颜色
                        "expand": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                        "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100, "step": 0.1}),
                    },
                    "optional": {
                        "all_mask": ("MASK",),
                        "mask": ("MASK",),
                    }
                }
    CATEGORY = "zdx/image"
    RETURN_TYPES = ("IMAGE", "MASK", "BOX",
                    "IMAGE", "MASK", "BOX", "INFO")
    RETURN_NAMES = ("croped_image", "croped_mask", "crop_box",
                    "scale_image", "scale_mask", "original_size", "info")
    FUNCTION = 'crop_by_mask_v2'
    DESCRIPTION = """
focuscrop V2 crop_by_mask_v2
"""
    @torch.inference_mode()     
    def crop_by_mask_v2(self, image, all_mask=None, up_keep=0.2, down_keep=0.2, right_keep=0.2, left_keep=0.2, crop_ratio="original" ,# is_focus=True, 
            mask=None,  aspect_ratio='original', expand=0, blur_radius=0.0, 
            fit='fill', method='lanczos', round_to_multiple=8, scale_to_side='longest', scale_to_length=1024, background_color='#000000',
        ):
        self.focus_crop = FocusCrop()
        self.growmaskwithblur = GrowMaskWithBlur()
        self.layerutility_imagescalebyaspectratio_v2 = ImageScaleByAspectRatioV2()

        if  mask is None:
            _, crop_h, crop_w, _ = image.shape
            return (image, mask, None, None, None, {})
        image_mask_cropbymask_1 =  self.focus_crop.crop_by_mask_v2(
            up_keep=up_keep, down_keep=down_keep, right_keep=right_keep, left_keep=left_keep,
            image=image,
            all_mask=all_mask,
            mask=mask,
            crop_ratio=crop_ratio,
        )
        info = image_mask_cropbymask_1[7]
        layerutility_imagescalebyaspectratio_v2_148 = (
            self.layerutility_imagescalebyaspectratio_v2.image_scale_by_aspect_ratio(
                aspect_ratio=aspect_ratio,
                proportional_width=1,
                proportional_height=1,
                fit=fit,
                method=method,
                round_to_multiple=round_to_multiple,
                scale_to_side=scale_to_side,
                scale_to_length=scale_to_length,
                background_color=background_color,
                image=image_mask_cropbymask_1[0],
                mask=image_mask_cropbymask_1[1],
            )
        )
        info['original_size'] = layerutility_imagescalebyaspectratio_v2_148[2]
        growmaskwithblur_315 = self.growmaskwithblur.expand_mask(
            expand=expand,
            incremental_expandrate=0,
            tapered_corners=True,
            flip_input=False,
            blur_radius=blur_radius,
            lerp_alpha=1,
            decay_factor=1,
            fill_holes=False,
            mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_148, 1),
        )
        return (
            image_mask_cropbymask_1[0], 
            image_mask_cropbymask_1[1], 
            image_mask_cropbymask_1[2],    # bbox
            layerutility_imagescalebyaspectratio_v2_148[0], growmaskwithblur_315[0],
            layerutility_imagescalebyaspectratio_v2_148[2],  # original_size
            info,
        )


class FocusCropRestore():
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "background_image": ("IMAGE", ),
                "croped_image": ("IMAGE", ),  #
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "crop_box": ("BOX",),
                "original_size": ("BOX",),
                "expand": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100, "step": 0.1}),
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "croped_mask": ("MASK",),
            }
        }

    CATEGORY = "zdx/image"
    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = 'restore_crop_box'
    DESCRIPTION = """
focuscrop restore restore_crop_box
"""
    @torch.inference_mode()     
    def restore_crop_box(self, background_image, croped_image, invert_mask=False, original_size=None, crop_box=None,
                         croped_mask=None, expand=0, blur_radius=0.0, fill_holes=False
                         ):
        imagescalerestorev2 = ImageScaleRestoreV2()
        restorecropbox = RestoreCropBox()
        growmaskwithblur = GrowMaskWithBlur()
        if original_size is None:
            return (croped_image, croped_mask, )
        imagescalerestorev2_ = imagescalerestorev2.image_scale_restore(
            scale=1,
            method="lanczos",
            scale_by="by_scale",
            scale_by_length=1024,
            image=croped_image,
            mask=croped_mask,
            original_size=original_size,
        )
        if crop_box is None:
            return (imagescalerestorev2_[0], imagescalerestorev2_[1], )
        final_mask = get_value_at_index(imagescalerestorev2_, 1)
        growmaskwithblur_315 = growmaskwithblur.expand_mask(
            expand=expand,
            incremental_expandrate=0,
            tapered_corners=True,
            flip_input=False,
            blur_radius=blur_radius,
            lerp_alpha=1,
            decay_factor=1,
            fill_holes=fill_holes,
            mask=final_mask,
        )
        real_w,real_h = crop_box[2]-crop_box[0], crop_box[3]-crop_box[1]
        orig_w,orig_h = imagescalerestorev2_[0].shape[2], imagescalerestorev2_[0].shape[1]
        # 余数，除不尽 ，相差一个像素
        extra_w, extra_h = (orig_w - real_w) % 2, (orig_h - real_h) % 2
        pad_w = (orig_w - real_w) // 2
        pad_h = (orig_h - real_h) // 2
        # unpad crop_box
        croped_image_unpad = get_value_at_index(imagescalerestorev2_, 0)
        croped_mask_unpad = get_value_at_index(growmaskwithblur_315, 0)
        if pad_w > 0 or pad_h > 0:
        # 2 restore pad
            # Handle 0 padding correctly (slice(0, -0) results in empty tensor)
            end_h = -pad_h-extra_h if pad_h > 0 else None
            end_w = -pad_w-extra_w if pad_w > 0 else None
            croped_image_unpad = imagescalerestorev2_[0][:, pad_h+extra_h:end_h, pad_w+extra_w:end_w, :]
            if croped_mask is not None:
                croped_mask_unpad = growmaskwithblur_315[0][:, pad_h+extra_h:end_h, pad_w+extra_w:end_w]

        restorecropbox_ = restorecropbox.restore_crop_box(
            invert_mask=invert_mask,
            background_image=background_image,
            croped_image=croped_image_unpad,
            crop_box=crop_box,
            croped_mask=croped_mask_unpad,
        )
        return restorecropbox_

class FocusCropRestoreUltra():
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "info": ("INFO",),
                "background_image": ("IMAGE", ),
                "croped_image": ("IMAGE", ),  #
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "expand": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100, "step": 0.1}),
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "croped_mask": ("MASK",),
            }
        }

    CATEGORY = "zdx/image"
    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = 'restore_crop_box'
    DESCRIPTION = """
focuscrop restore restore_crop_box
"""
    @torch.inference_mode()     
    def restore_crop_box(self, background_image, croped_image, invert_mask=False,info=None,
                         croped_mask=None, expand=0, blur_radius=0.0, fill_holes=False
                         ):
        imagescalerestorev2 = ImageScaleRestoreV2()
        restorecropbox = RestoreCropBox()
        growmaskwithblur = GrowMaskWithBlur()
        crop_box = info.get("crop_box", None)
        original_size = info.get("original_size", None)
        # 1 scale restore
        if original_size is None:
            return (croped_image, croped_mask, )
        imagescalerestorev2_ = imagescalerestorev2.image_scale_restore(
            scale=1,
            method="lanczos",
            scale_by="by_scale",
            scale_by_length=1024,
            image=croped_image,
            mask=croped_mask,
            original_size=original_size,
        )
        if crop_box is None:
            return (imagescalerestorev2_[0], imagescalerestorev2_[1], )

        pad = info.get("pad", None) 
        # 2 restore pad
        croped_image_unpad = get_value_at_index(imagescalerestorev2_, 0)
        croped_mask_unpad = get_value_at_index(imagescalerestorev2_, 1)
        if pad is not None:
            # Check if pad is new 4-element format or old 2-element format
            if len(pad) == 4:
                pad_left, pad_top, pad_right, pad_bottom = pad
            else:
                pad_left, pad_top = pad[0], pad[1]
                pad_right, pad_bottom = pad_left, pad_top # Fallback for symmetric padding

            # Handle 0 padding correctly (slice(0, -0) results in empty tensor)
            # Use specific padding values for each side
            end_h = -pad_bottom if pad_bottom > 0 else None
            end_w = -pad_right if pad_right > 0 else None
            
            # Start index is simply top/left padding
            start_h = pad_top
            start_w = pad_left
            
            croped_image_unpad = imagescalerestorev2_[0][:, start_h:end_h, start_w:end_w, :]
            if croped_mask is not None:
                croped_mask_unpad = imagescalerestorev2_[1][:, start_h:end_h, start_w:end_w]

        final_mask = croped_mask_unpad
        growmaskwithblur_315 = growmaskwithblur.expand_mask(
            expand=expand,
            incremental_expandrate=0,
            tapered_corners=True,
            flip_input=False,
            blur_radius=blur_radius,
            lerp_alpha=1,
            decay_factor=1,
            fill_holes=fill_holes,
            mask=final_mask,
        )
        restorecropbox_ = restorecropbox.restore_crop_box(
            invert_mask=invert_mask,
            background_image=background_image,
            croped_image=croped_image_unpad,
            crop_box=crop_box,
            croped_mask=growmaskwithblur_315[0],
        )
        return restorecropbox_

class DynamicAspectRatio():
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "image": ("IMAGE",),
                    },
                    "optional": {
                        "scale_image": ("IMAGE",),
                    }
                }
    CATEGORY = "zdx/image"
    RETURN_TYPES = (['original', 'custom', '1:1', '3:2', '4:3', '16:9', '2:3', '3:4', '9:16'], "INT", "INT",)
    RETURN_NAMES = ("aspect_ratio", "portional_with", "portional_height",)
    FUNCTION = 'dynamic_aspect_ratio'
    DESCRIPTION = """return aspect_ratio, proportional_width, proportional_height according to input image aspect ratio"""

    def dynamic_aspect_ratio(self, image, scale_image=None):
        _, h, w, _ = image.shape
        aspect_ratio_ = "original"
        proportional_w, proportional_h = 1, 1
        if scale_image is not None:
            _, scale_h, scale_w, _ = scale_image.shape
            proportional_w, proportional_h = scale_w, scale_h
        if w / h > 1.0:
            aspect_ratio_ = "custom"
        return (aspect_ratio_, proportional_w, proportional_h,)


class EditeMatch():
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "init_image": ("IMAGE",),
                        "init_mask": ("MASK",),
                        "reference_image": ("IMAGE",),
                        "reference_mask": ("MASK",),
                        "scale_rate": ("FLOAT", {"default": 0.0, "min": -1, "max": 1, "step": 0.01, }),
                    },
                    "optional": {
                        "max_or_min_match": (["max", "min"], {"default": "min"}),
                        "blend": ("BOOLEAN", {"default": True}),
                    }
                }
    CATEGORY = "zdx/image"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "FLOAT","FLOAT", "FLOAT")
    RETURN_NAMES = ("blend_image", "blend_mask", "white_image", "center_x", "center_y", "scale_size")
    FUNCTION = 'canvas_blend'
    DESCRIPTION = """
匹配编辑区域，其他区域留白处理
输入normlize init_image,init_mask,  and reference_image, reference_mask (rmbg or selected)
"""
    @torch.inference_mode()     
    def canvas_blend(self, init_image, init_mask, reference_image, reference_mask, scale_rate=0.0, max_or_min_match="min", blend=True):
        self.zdxmaskadd = MaskAdd()
        # focus enhance
        # self.layerutility_cropbymask_v2 = NODE_CLASS_MAPPINGS["LayerUtility: CropByMask V2"]()
        # self.layerutility_cropbymask = NODE_CLASS_MAPPINGS["LayerUtility: CropByMask"]()
        # self.layerutility_cropboxresolve = NODE_CLASS_MAPPINGS["LayerUtility: CropBoxResolve"]()
        # self.layerutility_imagescalebyaspectratio_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleByAspectRatio V2"]()
        # self.layerutility_imageblendadvance_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageBlendAdvance V2"]()
        # self.layerutility_colorimage_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ColorImage V2"]()
        self.imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        
        self.layerutility_cropbymask = CropByMask()
        self.layerutility_cropbymask_v2 = CropByMaskV2()
        self.layerutility_restorecropbox = RestoreCropBox()
        self.layerutility_imagescalebyaspectratio_v2 = ImageScaleByAspectRatioV2()
        self.layerutility_imageblendadvance_v2 = ImageBlendAdvanceV2()
        self.layerutility_imagescalerestore_v2 = ImageScaleRestore()
        self.layerutility_cropboxresolve = CropBoxResolve()
        self.layerutility_colorimage_v2 = ColorImageV2()

            
        if not blend or init_mask is None:
            return (init_image, init_mask, reference_image, 0.0, 0.0, 0.0)
        if not isinstance(init_image, tuple) and isinstance(init_image, torch.Tensor):
            init_image = (init_image, )
        if not isinstance(init_mask, tuple) and isinstance(init_mask, torch.Tensor):
            if len(init_mask.shape) > 3:
                init_mask = init_mask[0]
            init_mask = (init_mask, )
        if not isinstance(reference_image, tuple) and isinstance(reference_image, torch.Tensor):
            reference_image = (reference_image, )
        if not isinstance(reference_mask, tuple) and isinstance(reference_mask, torch.Tensor):
            if len(reference_mask.shape) > 3:
                reference_mask = reference_mask[0]
            reference_mask = (reference_mask, )

        source_mask_pil = tensor2pil(reference_mask[0]).convert('L')
        source_mask_bbox = source_mask_pil.getbbox()
        if source_mask_bbox is None:
            raise ValueError("No valid mask pixels found.")
        x1,y1,x2,y2 = source_mask_bbox
        src_obj_w, src_obj_h = x2 - x1, y2 - y1

        init_mask_pil = tensor2pil(init_mask[0]).convert('L')
        init_mask_bbox = init_mask_pil.getbbox()
        if init_mask_bbox is None:
            raise ValueError("No valid mask pixels found.")
        x1,y1,x2,y2 = init_mask_bbox
        init_obj_w, init_obj_h = x2 - x1, y2 - y1
        # 参考图 与 填充区域缩放比率
        scale_fit_ratio = min(init_obj_w / src_obj_w, init_obj_h / src_obj_h) if max_or_min_match == "min" else max(init_obj_w / src_obj_w, init_obj_h / src_obj_h)
        # 填充区域中心坐标 需要的是比例不是具体的值
        init_size_w, init_size_h = tensor2pil(get_value_at_index(init_image, 0)).size
        init_edit_area_x_center, init_edit_area_y_center = (x1 + init_obj_w / 2) * 100 / init_size_w, (y1 + init_obj_h / 2) * 100 / init_size_h
        
        ## for matching the paste area
        reference_cropbymask_1 = self.layerutility_cropbymask.crop_by_mask(
            invert_mask=False,
            detect="min_bounding_rect",
            top_reserve=0,
            bottom_reserve=0,
            left_reserve=0,
            right_reserve=0,
            image=get_value_at_index(reference_image, 0),
            mask_for_crop=get_value_at_index(reference_mask, 0),
        )
        # print(">> center from init_mask_bbox: ", center_x, center_y)
        print(">> center from init_area: ", init_edit_area_x_center, init_edit_area_y_center)

        ##  类似FastCavas 将参考图贴在目标区域~
        ref_fit_area_blendadvance_v2_1 = (
            self.layerutility_imageblendadvance_v2.image_blend_advance_v2(
                invert_mask=False,
                blend_mode="normal",
                opacity=100,
                x_percent=init_edit_area_x_center,
                y_percent=init_edit_area_y_center,
                mirror="None",
                scale=scale_fit_ratio + scale_rate,
                aspect_ratio=1,
                rotate=0,
                transform_method="lanczos",
                anti_aliasing=0,
                background_image=get_value_at_index( init_image, 0 ),  # init_目标图
                layer_image=get_value_at_index(reference_cropbymask_1, 0),
                layer_mask=get_value_at_index(reference_cropbymask_1, 1),
            )
        )
        ## ref_rm_paste white
        whilte_as_init_image = self.layerutility_colorimage_v2.color_image_v2(
            size="custom",
            custom_width=512,
            custom_height=512,
            color="#FFFFFF",
            size_as=get_value_at_index(ref_fit_area_blendadvance_v2_1, 0),
        )
        ref_imagecompositemasked_429 = self.imagecompositemasked.composite(
            x=0,
            y=0,
            resize_source=False,
            destination=get_value_at_index(whilte_as_init_image, 0),
            source=get_value_at_index(ref_fit_area_blendadvance_v2_1, 0),
            mask=get_value_at_index(ref_fit_area_blendadvance_v2_1, 1),
        )

        return (
            get_value_at_index(ref_fit_area_blendadvance_v2_1, 0),
            get_value_at_index(ref_fit_area_blendadvance_v2_1, 1),
            get_value_at_index(ref_imagecompositemasked_429, 0),
            init_edit_area_x_center, 
            init_edit_area_y_center,
            scale_fit_ratio + scale_rate,
        )


