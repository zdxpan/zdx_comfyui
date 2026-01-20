import torch
from .imagefunc import log, tensor2pil, pil2tensor, mask2image, gaussian_blur, min_bounding_rect, max_inscribed_rect, mask_area, num_round_up_to_multiple, draw_rect

NODE_NAME = 'CropByMask V2'

class CropByMaskV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        detect_mode = ['mask_area', 'min_bounding_rect', 'max_inscribed_rect']
        multiple_list = ['8', '16', '32', '64', '128', '256', '512', 'None']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "detect": (detect_mode,),
                "top_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "bottom_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "left_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "right_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "round_to_multiple": (multiple_list,),
            },
            "optional": {
                "crop_box": ("BOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "IMAGE",)
    RETURN_NAMES = ("croped_image", "croped_mask", "crop_box", "box_preview")
    FUNCTION = 'crop_by_mask_v2'
    CATEGORY = 'zdx/LayerUtility'

    def crop_by_mask_v2(self, image, mask, invert_mask, detect,
                     top_reserve, bottom_reserve,
                     left_reserve, right_reserve, round_to_multiple,
                     crop_box=None
                     ):

        # 1. Handle Mask (Support Batch)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        # Removed truncation of multiple masks to support video/batch inputs
        
        if invert_mask:
            mask = 1 - mask
            
        # 2. Detect Crop Box (if needed)
        # We need PIL image for detection functions and preview
        # Use first mask for detection/preview to avoid batch error
        mask_for_pil = mask[0] if mask.dim() == 3 else mask
        mask_pil = tensor2pil(mask_for_pil).convert('L')
        preview_image = mask_pil.convert('RGB')
        
        B, H, W, C = image.shape
        
        if crop_box is None:
            _mask_for_detect = mask2image(mask_for_pil)
            bluredmask = gaussian_blur(_mask_for_detect, 20).convert('L')
            
            x, y, w, h = 0, 0, 0, 0
            if detect == "min_bounding_rect":
                (x, y, w, h) = min_bounding_rect(bluredmask)
            elif detect == "max_inscribed_rect":
                (x, y, w, h) = max_inscribed_rect(bluredmask)
            else:
                (x, y, w, h) = mask_area(_mask_for_detect)
            
            x1 = x - left_reserve
            y1 = y - top_reserve
            x2 = x + w + right_reserve
            y2 = y + h + bottom_reserve
            
            # Initial clip to image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            
            if round_to_multiple != 'None':
                multiple = int(round_to_multiple)
                width = x2 - x1
                height = y2 - y1
                target_width = num_round_up_to_multiple(width, multiple)
                target_height = num_round_up_to_multiple(height, multiple)
                
                x1 -= (target_width - width) // 2
                y1 -= (target_height - height) // 2
                x2 = x1 + target_width
                y2 = y1 + target_height
            
            crop_box = (int(x1), int(y1), int(x2), int(y2))
            
            log(f"{NODE_NAME}: Box detected. x={x1},y={y1},width={x2-x1},height={y2-y1}")
            preview_image = draw_rect(preview_image, x, y, w, h, line_color="#F00000",
                                      line_width=(w + h) // 100)
        
        # Draw final crop box on preview
        preview_image = draw_rect(preview_image, crop_box[0], crop_box[1],
                                  crop_box[2] - crop_box[0], crop_box[3] - crop_box[1],
                                  line_color="#00F000",
                                  line_width=(crop_box[2] - crop_box[0] + crop_box[3] - crop_box[1]) // 200)

        # 3. Apply Crop with Padding
        x1, y1, x2, y2 = crop_box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - W)
        pad_b = max(0, y2 - H)
        
        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            # Pad Image: [B, H, W, C] -> permute -> pad -> permute
            img_p = image.permute(0, 3, 1, 2) # [B, C, H, W]
            img_p = torch.nn.functional.pad(img_p, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
            img_p = img_p.permute(0, 2, 3, 1) # [B, H+pt+pb, W+pl+pr, C]
            
            # Pad Mask: [1, H, W] -> unsqueeze -> pad -> squeeze
            mask_p = mask.unsqueeze(1) # [1, 1, H, W]
            mask_p = torch.nn.functional.pad(mask_p, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
            mask_p = mask_p.squeeze(1) # [1, H', W']
            
            # Update crop coords to refer to padded image
            x1 += pad_l
            y1 += pad_t
            x2 += pad_l
            y2 += pad_t
            
            crop_imgs = img_p[:, y1:y2, x1:x2, :]
            crop_mask = mask_p[:, y1:y2, x1:x2]
        else:
            # Clamp to ensure no crash if slightly off
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            crop_imgs = image[:, y1:y2, x1:x2, :]
            crop_mask = mask[:, y1:y2, x1:x2]

        if crop_mask.shape[0] == 1 and B > 1:
            crop_mask = crop_mask.repeat(B, 1, 1)

        log(f"{NODE_NAME} Processed {len(crop_imgs)} image(s).", message_type='finish')
        return (crop_imgs, crop_mask, list(crop_box), pil2tensor(preview_image),)

# 输入bbox,根据bbox 剪裁图像和mask（如果有） 
# 1、有 的时候输入的image和mask就是从一个视频读取得到的，基本上就是一批图像和mask ，他们的宽高应该是相等的
# 2、有 的时候输入的image和没有输入mask, 但是输入了一个bbox，这个时候输出crop_mask就是none
# 3、如果只输入IMAGE,没有输入mask,需要从mask里面 将所有的mask合并为一张，按照像素取并集，然后得到最小外接矩形 bbox
class CropByBBox:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
            }, "optional": {
                "mask": ("MASK",),
                "bbox": ("BOX",),
                "expand_ratio": ("FLOAT", {"default": 0.1, "min": 0, "max": 2.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "BOX",)
    RETURN_NAMES = ("croped_image", "croped_mask", "crop_box",)
    FUNCTION = 'crop_by_bbox'
    CATEGORY = 'zdx/LayerUtility'

    def crop_by_bbox(self, image, mask=None, bbox=None, expand_ratio=0.1):
        _, H, W, _ = image.shape
        
        if bbox is None:
            if mask is not None:
                # Calculate bbox from mask union
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                
                # Union of all masks in batch
                combined_mask = torch.max(mask, dim=0)[0] # [H, W]
                
                nonzero = torch.nonzero(combined_mask > 0.0)
                if nonzero.numel() == 0:
                    x1, y1, x2, y2 = 0, 0, W, H
                else:
                    y1 = nonzero[:, 0].min().item()
                    y2 = nonzero[:, 0].max().item() + 1
                    x1 = nonzero[:, 1].min().item()
                    x2 = nonzero[:, 1].max().item() + 1

                    if expand_ratio > 0:
                        box_w = x2 - x1
                        box_h = y2 - y1
                        pad_w = int(box_w * expand_ratio * 0.5)
                        pad_h = int(box_h * expand_ratio * 0.5)
                        x1 -= pad_w
                        y1 -= pad_h
                        x2 += pad_w
                        y2 += pad_h
            else:
                # No bbox, no mask. Return original.
                x1, y1, x2, y2 = 0, 0, W, H
        else:
            x1, y1, x2, y2 = bbox
            
        # Ensure coordinates are within bounds
        x1 = max(0, min(int(x1), W))
        y1 = max(0, min(int(y1), H))
        x2 = max(0, min(int(x2), W))
        y2 = max(0, min(int(y2), H))
        
        if x1 >= x2 or y1 >= y2:
             log(f"CropByBBox: Invalid bbox ({x1},{y1},{x2},{y2}), returning original image.", message_type='warning')
             x1, y1, x2, y2 = 0, 0, W, H

        crop_box = (x1, y1, x2, y2)
        
        # Crop image
        croped_image = image[:, y1:y2, x1:x2, :]
        
        croped_mask = None
        if mask is not None:
             if mask.dim() == 2:
                 mask = mask.unsqueeze(0)
             croped_mask = mask[:, y1:y2, x1:x2]
        
        return (croped_image, croped_mask, list(crop_box),)
