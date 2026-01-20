import torch
from .imagefunc import log

NODE_NAME = 'RestoreCropBox'


def unpad_restore(image, mask, pad):
    # 2 restore pad
    croped_image_unpad = image
    croped_mask_unpad = mask
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
        
        croped_image_unpad = croped_image_unpad[:, start_h:end_h, start_w:end_w, :]
        if croped_mask_unpad is not None:
            croped_mask_unpad = croped_mask_unpad[:, start_h:end_h, start_w:end_w]
    return (croped_image_unpad, croped_mask_unpad, )

# docstring
"""
RestoreCropBox
- 如果你有 100张 background_image （视频背景）和 100张 croped_image （修复后的人脸），它会一一对应处理。
- 如果你有 1张 background_image （静态背景）和 100张 croped_image ，它会将这一张背景重复使用100次。
- 如果你有 100张 croped_image 但只有 1个 crop_box （如你所述的合并Mask后的固定BBox），它会将每一张小图都贴到同一个位置。
"""

class RestoreCropBox:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "background_image": ("IMAGE", ),
                "croped_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "crop_box": ("BOX",),
            },
            "optional": {
                "croped_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = 'restore_crop_box'
    CATEGORY = 'zdx/LayerUtility'

    def restore_crop_box(self, background_image, croped_image, invert_mask, crop_box,
                         croped_mask=None
                         ):
        
        # Ensure inputs are tensors [B, H, W, C]
        if background_image.dim() == 3:
            background_image = background_image.unsqueeze(0)
        if croped_image.dim() == 3:
            croped_image = croped_image.unsqueeze(0)
            
        x1, y1, x2, y2 = crop_box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        box_w = x2 - x1
        box_h = y2 - y1
        
        # 1. Unpad Logic (Center Crop if image is larger than box)
        orig_h, orig_w = croped_image.shape[1], croped_image.shape[2]
        
        # Calculate margins
        margin_w = max(0, orig_w - box_w)
        margin_h = max(0, orig_h - box_h)
        
        # Only unpad if positive padding (image > box)
        if margin_w > 0 or margin_h > 0:
            print(f'pad_h,pad_w {margin_h},{margin_w}, into unpad operation!!!')
            
            # Center crop logic: 
            # If diff is 1, start=0. (Keep 0..box_w). Drops last pixel.
            # If diff is 2, start=1. (Keep 1..box_w+1). Drops first and last.
            start_h = margin_h // 2
            start_w = margin_w // 2
            
            # Use 'pad' variable naming to match previous logic logic for compatibility if needed, 
            # but strictly define end based on box size
            end_h = start_h + box_h
            end_w = start_w + box_w
            
            # Clamp to original image bounds just in case (though margin check ensures start is valid)
            end_h = min(end_h, orig_h)
            end_w = min(end_w, orig_w)
            
            croped_image = croped_image[:, start_h:end_h, start_w:end_w, :]
            
            if croped_mask is not None:
                if croped_mask.dim() == 2:
                     croped_mask = croped_mask.unsqueeze(0)
                # If mask is [B, H, W], slice H, W (dim 1, 2)
                croped_mask = croped_mask[:, start_h:end_h, start_w:end_w]

        # 2. Prepare Mask
        if croped_mask is None:
            if croped_image.shape[-1] == 4:
                # Extract Alpha [B, H, W, 1]
                mask_sub = croped_image[:, :, :, 3].unsqueeze(-1)
                croped_image = croped_image[:, :, :, :3] # Keep RGB
            else:
                # White mask [B, H, W, 1]
                mask_sub = torch.ones((croped_image.shape[0], croped_image.shape[1], croped_image.shape[2], 1), 
                                      device=croped_image.device, dtype=croped_image.dtype)
        else:
            if croped_mask.dim() == 2: 
                croped_mask = croped_mask.unsqueeze(0)
            # Ensure [B, H, W, 1]
            if croped_mask.dim() == 3:
                mask_sub = croped_mask.unsqueeze(-1)
            else:
                mask_sub = croped_mask

        if invert_mask:
            mask_sub = 1.0 - mask_sub
            
        # 3. Broadcast Batches (Replicating the "repeat last" logic efficiently)
        B_bg = background_image.shape[0]
        B_fg = croped_image.shape[0]
        B_m = mask_sub.shape[0]
        B_max = max(B_bg, B_fg, B_m)
        
        # Helper to broadcast by repeating last element
        def broadcast_tensor(tensor, target_len):
            current_len = tensor.shape[0]
            if current_len == target_len:
                return tensor
            # Create indices: [0, 1, ..., N-1, N-1, N-1]
            indices = torch.arange(target_len, device=tensor.device)
            indices = torch.clamp(indices, max=current_len - 1)
            return tensor[indices]

        background_image = broadcast_tensor(background_image, B_max)
        croped_image = broadcast_tensor(croped_image, B_max)
        mask_sub = broadcast_tensor(mask_sub, B_max)

        # 4. Composite
        out_image = background_image.clone()
        out_mask = torch.zeros((B_max, background_image.shape[1], background_image.shape[2]), 
                               device=background_image.device, dtype=background_image.dtype)
                               
        tx, ty = x1, y1
        sh, sw = croped_image.shape[1], croped_image.shape[2]
        H_bg, W_bg = background_image.shape[1], background_image.shape[2]
        
        # Compute valid intersection area to avoid out-of-bounds
        bg_y1 = max(0, ty)
        bg_x1 = max(0, tx)
        bg_y2 = min(H_bg, ty + sh)
        bg_x2 = min(W_bg, tx + sw)
        
        if bg_y1 < bg_y2 and bg_x1 < bg_x2:
            src_y1 = bg_y1 - ty
            src_x1 = bg_x1 - tx
            src_y2 = src_y1 + (bg_y2 - bg_y1)
            src_x2 = src_x1 + (bg_x2 - bg_x1)
            
            source_slice = croped_image[:, src_y1:src_y2, src_x1:src_x2, :]
            mask_slice = mask_sub[:, src_y1:src_y2, src_x1:src_x2, :]
            bg_slice = out_image[:, bg_y1:bg_y2, bg_x1:bg_x2, :]
            
            # Alpha Blend: Output = BG * (1 - Mask) + FG * Mask
            blended = bg_slice * (1.0 - mask_slice) + source_slice * mask_slice
            out_image[:, bg_y1:bg_y2, bg_x1:bg_x2, :] = blended
            out_mask[:, bg_y1:bg_y2, bg_x1:bg_x2] = mask_slice.squeeze(-1)
            
        log(f"{NODE_NAME} Processed {B_max} image(s).", message_type='finish')
        return (out_image, out_mask,)

class RestoreCropBoxPad(RestoreCropBox):
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "background_image": ("IMAGE", ),
                "croped_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "info": ("INFO",),
            },
            "optional": {
                "croped_mask": ("MASK",),
                "crop_box": ("BOX",),
            }
        }
    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = 'restore_crop_box'
    CATEGORY = 'zdx/LayerUtility'
    @torch.inference_mode()
    def restore_crop_box(self, background_image, croped_image, invert_mask, crop_box=None,
                         croped_mask=None, info={}):
        pad = info.get("pad", None)
        if crop_box is None:
            crop_box = info.get("crop_box", None)
        if crop_box is None:
            raise ValueError("please use FocusCrop to get crop_box and pad info")

        # 2 restore pad
        croped_image_unpad, croped_mask_unpad = unpad_restore(croped_image, croped_mask, pad)
        
        # Adjust crop_box to match the unpadded image
        if pad is not None:
            if len(pad) == 4:
                pad_left, pad_top, pad_right, pad_bottom = pad
            else:
                pad_left, pad_top = pad[0], pad[1]
                pad_right, pad_bottom = pad_left, pad_top
            
            x1, y1, x2, y2 = crop_box
            crop_box = (x1 + pad_left, y1 + pad_top, x2 - pad_right, y2 - pad_bottom)

        return super().restore_crop_box(background_image=background_image, croped_image=croped_image_unpad, 
                invert_mask=invert_mask, crop_box=crop_box, croped_mask=croped_mask_unpad)

