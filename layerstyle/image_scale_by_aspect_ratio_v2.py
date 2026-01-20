import torch
from PIL import Image
import math
from .imagefunc import log, tensor2pil, pil2tensor, image2mask, num_round_up_to_multiple, fit_resize_image, is_valid_mask



class ImageScaleByAspectRatioV2:

    def __init__(self):
        self.NODE_NAME = 'ImageScaleByAspectRatio V2'

    @classmethod
    def INPUT_TYPES(self):
        ratio_list = ['original', 'custom', '1:1', '3:2', '4:3', '16:9', '2:3', '3:4', '9:16']
        fit_mode = ['letterbox', 'crop', 'fill']
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        scale_to_list = ['None', 'longest', 'shortest', 'width', 'height', 'total_pixel(kilo pixel)']
        return {
            "required": {
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
                "image": ("IMAGE",),  #
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "INT", "INT",)
    RETURN_NAMES = ("image", "mask", "original_size", "width", "height",)
    FUNCTION = 'image_scale_by_aspect_ratio'
    CATEGORY = 'zdx/LayerUtility'

    def image_scale_by_aspect_ratio(self, aspect_ratio, proportional_width, proportional_height,
                                    fit, method, round_to_multiple, scale_to_side, scale_to_length,
                                    background_color,
                                    image=None, mask = None,
                                    ):
        orig_width = 0
        orig_height = 0
        target_width = 0
        target_height = 0
        ratio = 1.0
        ret_images = None
        ret_masks = None

        # 1. Determine Original Size
        if image is not None:
            orig_width = image.shape[2]
            orig_height = image.shape[1]
        
        if mask is not None:
            if image is None: # If only mask is provided, use its size
                orig_width = mask.shape[2] if mask.dim() == 3 else mask.shape[1]
                orig_height = mask.shape[1] if mask.dim() == 3 else mask.shape[0]
            else:
                # Check mismatch if both exist
                m_w = mask.shape[2] if mask.dim() == 3 else mask.shape[1]
                m_h = mask.shape[1] if mask.dim() == 3 else mask.shape[0]
                if (orig_width > 0 and orig_width != m_w) or (orig_height > 0 and orig_height != m_h):
                    log(f"Error: {self.NODE_NAME} execute failed, because the mask is does'nt match image.", message_type='error')
                    return (None, None, None, 0, 0,)

        if orig_width == 0 and orig_height == 0:
            log(f"Error: {self.NODE_NAME} execute failed, because the image or mask at least one must be input.", message_type='error')
            return (None, None, None, 0, 0,)

        # 2. Calculate Target Ratio
        if aspect_ratio == 'original':
            ratio = orig_width / orig_height
        elif aspect_ratio == 'custom':
            ratio = proportional_width / proportional_height
        else:
            s = aspect_ratio.split(":")
            ratio = int(s[0]) / int(s[1])

        # 3. Calculate Target Dimensions
        if ratio > 1:
            if scale_to_side == 'longest':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'shortest':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'width':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'height':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'total_pixel(kilo pixel)':
                target_width = math.sqrt(ratio * scale_to_length * 1000)
                target_height = target_width / ratio
                target_width = int(target_width)
                target_height = int(target_height)
            else:
                target_width = orig_width
                target_height = int(target_width / ratio)
        else:
            if scale_to_side == 'longest':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'shortest':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'width':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'height':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'total_pixel(kilo pixel)':
                target_width = math.sqrt(ratio * scale_to_length * 1000)
                target_height = target_width / ratio
                target_width = int(target_width)
                target_height = int(target_height)
            else:
                target_height = orig_height
                target_width = int(target_height * ratio)

        if round_to_multiple != 'None':
            multiple = int(round_to_multiple)
            target_width = num_round_up_to_multiple(target_width, multiple)
            target_height = num_round_up_to_multiple(target_height, multiple)

        # 4. Processing
        import torch.nn.functional as F

        # Helper: Resize logic
        def resize_tensor(tensor, mode='bilinear', is_mask=False):
            # tensor: [B, C, H, W]
            if fit == 'letterbox':
                _, _, h, w = tensor.shape
                scale = min(target_width/w, target_height/h)
                new_h, new_w = int(h * scale), int(w * scale)
                
                # Resize
                resized = F.interpolate(tensor, size=(new_h, new_w), mode=mode, align_corners=False if mode != 'nearest' else None)
                
                # Create canvas
                if is_mask:
                     canvas = torch.zeros((tensor.shape[0], 1, target_height, target_width), device=tensor.device)
                else:
                     # Parse background color
                     bg_color_hex = background_color.lstrip('#')
                     bg_rgb = [int(bg_color_hex[i:i+2], 16)/255.0 for i in (0, 2, 4)]
                     canvas = torch.tensor(bg_rgb, device=tensor.device).view(1, 3, 1, 1)
                     canvas = canvas.repeat(tensor.shape[0], 1, target_height, target_width)
                
                # Paste center
                y1 = (target_height - new_h) // 2
                x1 = (target_width - new_w) // 2
                canvas[:, :, y1:y1+new_h, x1:x1+new_w] = resized
                return canvas

            elif fit == 'crop':
                _, _, h, w = tensor.shape
                target_ratio = target_width / target_height
                curr_ratio = w / h
                
                if curr_ratio > target_ratio: # Wider
                    w_crop = int(h * target_ratio)
                    x1 = (w - w_crop) // 2
                    cropped = tensor[:, :, :, x1:x1+w_crop]
                else: # Taller
                    h_crop = int(w / target_ratio)
                    y1 = (h - h_crop) // 2
                    cropped = tensor[:, :, y1:y1+h_crop, :]
                
                return F.interpolate(cropped, size=(target_height, target_width), mode=mode, align_corners=False if mode != 'nearest' else None)
            
            else: # fill
                return F.interpolate(tensor, size=(target_height, target_width), mode=mode, align_corners=False if mode != 'nearest' else None)

        # Map method
        interp_method = 'nearest'
        if method in ['bicubic', 'lanczos']:
            interp_method = 'bicubic'
        elif method in ['bilinear', 'hamming', 'box']:
            interp_method = 'bilinear'

        if image is not None:
            # [B, H, W, C] -> [B, C, H, W]
            img_batch = image.movedim(-1, 1)
            ret_images = resize_tensor(img_batch, mode=interp_method, is_mask=False)
            # [B, C, H, W] -> [B, H, W, C]
            ret_images = ret_images.movedim(1, -1)
            
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            # [B, H, W] -> [B, 1, H, W]
            mask_batch = mask.unsqueeze(1)
            # Use same method or nearest? Original code uses same method for mask.
            ret_masks = resize_tensor(mask_batch, mode=interp_method, is_mask=True)
            # [B, 1, H, W] -> [B, H, W]
            ret_masks = ret_masks.squeeze(1)

        if ret_images is not None:
            log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        elif ret_masks is not None:
            log(f"{self.NODE_NAME} Processed {len(ret_masks)} mask(s).", message_type='finish')

        return (ret_images, ret_masks, [orig_width, orig_height], target_width, target_height,)
