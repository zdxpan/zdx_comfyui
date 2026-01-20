import torch
import numpy as np
from PIL import Image
import math
import nodes
import cv2
from collections import Counter


class MaskCropV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_fill": ("BOOLEAN", {"default": False}),
                "expansion_top": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 3.0, "step": 0.1}),
                "expansion_bottom": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 3.0, "step": 0.1}),
                "expansion_left": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 3.0, "step": 0.1}),
                "expansion_right": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 3.0, "step": 0.1}),
                "output_size": (["Original", "Original 1:1", "Custom Width/Height"], {"default": "Original"}),
                "custom_width": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "custom_height": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "multiple_round": ("INT", {"default": 8, "min": 0, "max": 256}),
            }
        }
    
    RETURN_TYPES = ("SEAM", "IMAGE", "MASK")
    RETURN_NAMES = ("seam", "cropped_image", "cropped_mask")
    FUNCTION = "crop_image"
    CATEGORY = "zdx/mask"
    
    def fill_mask_holes(self, mask_np):
        """填充遮罩内部的孔洞"""
        from scipy import ndimage
        
        # 二值化遮罩
        binary_mask = mask_np > 0.1
        
        # 使用形态学操作填充孔洞
        filled_mask = ndimage.binary_fill_holes(binary_mask)
        
        # 转换回原始范围
        filled_mask = filled_mask.astype(np.float32)
        
        return filled_mask
    
    def calculate_multiple_round(self, value, multiple):
        """修改为向下取整"""
        if multiple <= 1:
            return value
        return math.floor(value / multiple) * multiple
    
    def get_mode_color(self, image, region, direction, sample_width=10):
        """获取指定区域边缘颜色的众数（出现频率最高的颜色）"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        w, h = image.size
        left, top, right, bottom = region
        
        # 定义采样区域
        if direction == "left" and left < 0:
            # 取最左侧的一列像素进行采样
            sample_left = 0
            sample_right = min(sample_width, w)
            sample_top = max(0, top)
            sample_bottom = min(h, bottom)
        elif direction == "right" and right > w:
            # 取最右侧的一列像素进行采样
            sample_right = w
            sample_left = max(0, w - sample_width)
            sample_top = max(0, top)
            sample_bottom = min(h, bottom)
        elif direction == "top" and top < 0:
            # 取最上方的一行像素进行采样
            sample_top = 0
            sample_bottom = min(sample_width, h)
            sample_left = max(0, left)
            sample_right = min(w, right)
        elif direction == "bottom" and bottom > h:
            # 取最下方的一行像素进行采样
            sample_bottom = h
            sample_top = max(0, h - sample_width)
            sample_left = max(0, left)
            sample_right = min(w, right)
        else:
            # 不需要填充或方向无效，返回黑色
            return (0, 0, 0)
        
        # 提取采样区域的像素
        if sample_right > sample_left and sample_bottom > sample_top:
            sample_region = image.crop((sample_left, sample_top, sample_right, sample_bottom))
            pixel_data = list(sample_region.getdata())
            
            if pixel_data:
                # 对颜色进行量化（将相近颜色归为一类）
                quantized_pixels = []
                for r, g, b in pixel_data:
                    # 将颜色量化为16级，减少颜色种类
                    quantized_r = (r // 16) * 16
                    quantized_g = (g // 16) * 16
                    quantized_b = (b // 16) * 16
                    quantized_pixels.append((quantized_r, quantized_g, quantized_b))
                
                # 计算众数颜色
                color_counts = Counter(quantized_pixels)
                mode_color = color_counts.most_common(1)[0][0]
                return mode_color
        
        return (0, 0, 0)
    
    def get_edge_padding_image(self, image, target_region, original_width, original_height, is_mask=False):
        """使用众数颜色进行填充，每个需要填充的方向都使用该方向的边缘众数颜色填充"""
        left, top, right, bottom = target_region
        
        # 计算实际需要的尺寸
        target_width = right - left
        target_height = bottom - top
        
        # 创建目标图像
        if is_mask:
            # 对于遮罩，超出部分始终用黑色填充
            target_image = Image.new("L", (target_width, target_height), 0)
        else:
            # 对于图像，使用原图模式
            target_image = Image.new(image.mode, (target_width, target_height))
            
            # 预计算各方向的众数颜色
            left_mode = self.get_mode_color(image, (left, top, right, bottom), "left")
            right_mode = self.get_mode_color(image, (left, top, right, bottom), "right")
            top_mode = self.get_mode_color(image, (left, top, right, bottom), "top")
            bottom_mode = self.get_mode_color(image, (left, top, right, bottom), "bottom")
        
        # 计算在原始图像内的有效区域
        valid_left = max(0, left)
        valid_top = max(0, top)
        valid_right = min(original_width, right)
        valid_bottom = min(original_height, bottom)
        
        # 计算在目标图像中的对应位置
        target_left_offset = valid_left - left
        target_top_offset = valid_top - top
        
        # 如果有有效区域，从原图复制
        if valid_right > valid_left and valid_bottom > valid_top:
            valid_region = image.crop((valid_left, valid_top, valid_right, valid_bottom))
            target_image.paste(valid_region, (target_left_offset, target_top_offset))
        
        # 对于图像，使用众数颜色填充超出边界的部分
        if not is_mask:
            # 判断需要填充的方向（每个方向独立判断，不再限制只填充一个方向）
            need_fill_left = left < 0
            need_fill_right = right > original_width
            need_fill_top = top < 0
            need_fill_bottom = bottom > original_height
            
            # 左侧填充
            if need_fill_left:
                fill_width = -left
                for x in range(fill_width):
                    for y in range(target_height):
                        target_image.putpixel((x, y), left_mode)
            
            # 右侧填充
            if need_fill_right:
                fill_width = right - original_width
                start_x = target_width - fill_width
                for x in range(start_x, target_width):
                    for y in range(target_height):
                        target_image.putpixel((x, y), right_mode)
            
            # 上方填充
            if need_fill_top:
                fill_height = -top
                for y in range(fill_height):
                    for x in range(target_width):
                        target_image.putpixel((x, y), top_mode)
            
            # 下方填充
            if need_fill_bottom:
                fill_height = bottom - original_height
                start_y = target_height - fill_height
                for y in range(start_y, target_height):
                    for x in range(target_width):
                        target_image.putpixel((x, y), bottom_mode)
        
        return target_image
    
    def crop_image(self, image, mask, mask_fill, expansion_top, expansion_bottom, expansion_left, expansion_right, 
                  output_size, custom_width, custom_height, multiple_round):
        # Ensure input is single image
        if image.shape[0] > 1:
            image = image[0:1]
        if mask.shape[0] > 1:
            mask = mask[0:1]
            
        # Convert tensor to numpy array
        image_np = image[0].cpu().numpy()
        mask_np = mask[0].cpu().numpy()
        
        # Save filled mask to seam data
        filled_mask_np = mask_np.copy()
        
        # Handle mask based on mask_fill toggle
        if mask_fill:
            filled_mask_np = self.fill_mask_holes(mask_np)
        
        # Get original dimensions
        original_height, original_width = image_np.shape[:2]
        
        # Find non-zero region boundaries (using filled mask)
        non_zero_region = np.where(filled_mask_np > 0.1)  # Use threshold to avoid noise
        if len(non_zero_region[0]) == 0:
            # If no mask region, use entire image
            min_y, max_y, min_x, max_x = 0, original_height, 0, original_width
            mask_width = original_width
            mask_height = original_height
        else:
            min_y, max_y = np.min(non_zero_region[0]), np.max(non_zero_region[0])
            min_x, max_x = np.min(non_zero_region[1]), np.max(non_zero_region[1])
            mask_width = max_x - min_x
            mask_height = max_y - min_y
        
        # Calculate mask shortest side
        mask_shortest_side = min(mask_width, mask_height)
        
        # Calculate expansion amount for four directions (multiply by mask shortest side * (expansion coefficient - 1))
        expand_top = int(mask_shortest_side * (expansion_top - 1.0))
        expand_bottom = int(mask_shortest_side * (expansion_bottom - 1.0))
        expand_left = int(mask_shortest_side * (expansion_left - 1.0))
        expand_right = int(mask_shortest_side * (expansion_right - 1.0))
        
        # Limit expansion to not exceed image boundaries
        expand_top = min(expand_top, min_y)  # Upward expansion cannot exceed image top
        expand_bottom = min(expand_bottom, original_height - max_y - 1)  # Downward expansion cannot exceed image bottom
        expand_left = min(expand_left, min_x)  # Leftward expansion cannot exceed image left
        expand_right = min(expand_right, original_width - max_x - 1)  # Rightward expansion cannot exceed image right
        
        # Calculate base crop region (mask region + expansion amount)
        base_top = min_y - expand_top
        base_bottom = max_y + expand_bottom
        base_left = min_x - expand_left
        base_right = max_x + expand_right
        
        # Determine final crop region based on output size option
        if output_size == "Original":
            # Original: Mask + expansion, then multiple round
            crop_width = base_right - base_left
            crop_height = base_bottom - base_top
            
            if multiple_round > 1:
                crop_width = self.calculate_multiple_round(crop_width, multiple_round)
                crop_height = self.calculate_multiple_round(crop_height, multiple_round)
            
            # Keep center point unchanged, adjust size
            center_x = (base_left + base_right) // 2
            center_y = (base_top + base_bottom) // 2
            
            final_left = center_x - crop_width // 2
            final_right = final_left + crop_width
            final_top = center_y - crop_height // 2
            final_bottom = final_top + crop_height
            
        elif output_size == "Original 1:1":
            # Original 1:1: Take minimum bounding square, then multiple round
            base_width = base_right - base_left
            base_height = base_bottom - base_top
            target_size = max(base_width, base_height)
            
            if multiple_round > 1:
                target_size = self.calculate_multiple_round(target_size, multiple_round)
            
            # Keep center point unchanged, adjust size
            center_x = (base_left + base_right) // 2
            center_y = (base_top + base_bottom) // 2
            
            final_left = center_x - target_size // 2
            final_right = final_left + target_size
            final_top = center_y - target_size // 2
            final_bottom = final_top + target_size
            
        else:  # Custom Width/Height
            # First round user input
            target_width = custom_width
            target_height = custom_height
            if multiple_round > 1:
                target_width = self.calculate_multiple_round(target_width, multiple_round)
                target_height = self.calculate_multiple_round(target_height, multiple_round)
            
            # Calculate base aspect ratio and target aspect ratio
            base_width = base_right - base_left
            base_height = base_bottom - base_top
            base_aspect_ratio = base_width / base_height
            target_aspect_ratio = target_width / target_height
            
            # Determine expansion direction
            if base_aspect_ratio < target_aspect_ratio:
                # Need to expand width
                expanded_width = int(base_height * target_aspect_ratio)
                expanded_height = base_height
            else:
                # Need to expand height
                expanded_width = base_width
                expanded_height = int(base_width / target_aspect_ratio)
            
            # Keep center point unchanged, adjust size
            center_x = (base_left + base_right) // 2
            center_y = (base_top + base_bottom) // 2
            
            final_left = center_x - expanded_width // 2
            final_right = final_left + expanded_width
            final_top = center_y - expanded_height // 2
            final_bottom = final_top + expanded_height
        
        # Convert to integer coordinates
        final_left = int(final_left)
        final_right = int(final_right)
        final_top = int(final_top)
        final_bottom = int(final_bottom)
        
        # Calculate actual crop dimensions
        crop_width = final_right - final_left
        crop_height = final_bottom - final_top
        
        # Convert to PIL image for processing
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_mask = Image.fromarray((filled_mask_np * 255).astype(np.uint8))
        
        # Get crop region (using new padding method)
        cropped_image = self.get_edge_padding_image(pil_image, (final_left, final_top, final_right, final_bottom), original_width, original_height, is_mask=False)
        cropped_mask = self.get_edge_padding_image(pil_mask, (final_left, final_top, final_right, final_bottom), original_width, original_height, is_mask=True)
        
        # Save unscaled cropped image and mask, used for calculating position during restoration
        unscaled_cropped_image = cropped_image.copy()
        unscaled_crop_mask = cropped_mask.copy()
        
        # For Custom Width/Height mode, need to scale proportionally to target size
        if output_size == "Custom Width/Height":
            current_width, current_height = cropped_image.size
            scale_ratio_width = target_width / current_width
            scale_ratio_height = target_height / current_height
            scale_ratio = min(scale_ratio_width, scale_ratio_height)
            
            scaled_width = int(current_width * scale_ratio)
            scaled_height = int(current_height * scale_ratio)
            
            cropped_image = cropped_image.resize((scaled_width, scaled_height), Image.LANCZOS)
            cropped_mask = cropped_mask.resize((scaled_width, scaled_height), Image.LANCZOS)
            
            # If needed, fill to target size
            if scaled_width != target_width or scaled_height != target_height:
                # Create target size image, filled with black
                final_image = Image.new(cropped_image.mode, (target_width, target_height), 0)
                final_mask = Image.new(cropped_mask.mode, (target_width, target_height), 0)
                
                # Calculate paste position (centered)
                left_offset = (target_width - scaled_width) // 2
                top_offset = (target_height - scaled_height) // 2
                
                final_image.paste(cropped_image, (left_offset, top_offset))
                final_mask.paste(cropped_mask, (left_offset, top_offset))
                
                cropped_image = final_image
                cropped_mask = final_mask
                
                # Save padding info
                padding_info = (left_offset, top_offset, scaled_width, scaled_height, target_width, target_height)
            else:
                padding_info = None
        else:
            target_width, target_height = cropped_image.size
            padding_info = None
        
        # Convert back to numpy array
        final_image_np = np.array(cropped_image).astype(np.float32) / 255.0
        final_mask_np = np.array(cropped_mask).astype(np.float32) / 255.0
        
        # Convert to tensor
        final_image_tensor = torch.from_numpy(final_image_np)[None, ...]
        final_mask_tensor = torch.from_numpy(final_mask_np)[None, ...]
        
        # Calculate scale ratio (for restoration)
        if output_size == "Custom Dimensions":
            # In Custom Dimensions mode, there are two scales: from crop region to expanded region, then to target size
            original_crop_width = final_right - final_left
            original_crop_height = final_bottom - final_top
            expanded_width = unscaled_cropped_image.size[0]
            expanded_height = unscaled_cropped_image.size[1]
            
            scale_ratio_width = expanded_width / original_crop_width
            scale_ratio_height = expanded_height / original_crop_height
            final_scale_width = target_width / expanded_width if output_size == "Custom Dimensions" else 1.0
            final_scale_height = target_height / expanded_height if output_size == "Custom Dimensions" else 1.0
        else:
            # In other modes, only one scale: from crop region to target size
            original_crop_width = final_right - final_left
            original_crop_height = final_bottom - final_top
            scale_ratio_width = target_width / original_crop_width
            scale_ratio_height = target_height / original_crop_height
            final_scale_width = 1.0
            final_scale_height = 1.0
        
        # Create seam data (containing all necessary information)
        seam_data = {
            "original_image": image,  # original image tensor
            "original_mask": mask,    # original mask tensor
            "filled_mask": torch.from_numpy(filled_mask_np)[None, ...],  
            "original_size": (original_height, original_width),
            "crop_region": (final_top, final_bottom, final_left, final_right),
            "target_size": (target_height, target_width),
            "scale_ratio_width": scale_ratio_width,
            "scale_ratio_height": scale_ratio_height,
            "final_scale_width": final_scale_width,
            "final_scale_height": final_scale_height,
            "output_size_mode": output_size,
            "padding_info": padding_info,  # Save padding info for restoration
            "mask_fill": mask_fill, 
            "mask_white_region_width": mask_width, 
            "mask_white_region_height": mask_height,
            "unscaled_crop_mask": unscaled_crop_mask  # Save unscaled crop mask for restoration
        }
        
        return (seam_data, final_image_tensor, final_mask_tensor)


class CropRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seam": ("SEAM",),
                "cropped_image": ("IMAGE",),
                "mask_expansion_percentage": ("INT", {"default": 5, "min": 0, "max": 30, "step": 1}),
                "mask_feathering_percentage": ("INT", {"default": 5, "min": 0, "max": 30, "step": 1}),
            },
            "optional": {
                "cropped_mask": ("MASK",),
                "background_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore_image"
    CATEGORY = "zdx/mask"
    
    def tensor2pil(self, image):
        """将tensor转换为PIL图像"""
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
    def pil2tensor(self, image):
        """将PIL图像转换为tensor"""
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
    
    def image_to_rgb(self, images):
        """将图像转换为RGB格式"""
        if len(images) > 1:
            tensors = []
            for image in images:
                pil_image = self.tensor2pil(image)
                pil_image = pil_image.convert('RGB')
                tensors.append(self.pil2tensor(pil_image))
            tensors = torch.cat(tensors, dim=0)
            return (tensors, )
        else:
            pil_image = self.tensor2pil(images)
            pil_image = pil_image.convert('RGB')
            return (self.pil2tensor(pil_image), )
    
    def calculate_percentage_pixel_value(self, percentage, mask_width, mask_height):
        """Convert percentage to pixel value, based on average of mask white region width and height"""
        if percentage == 0:
            return 0
        average = (mask_width + mask_height) / 2
        pixel_value = int(average * percentage / 100)
        return max(1, pixel_value)  # Ensure at least 1 pixel, avoid 0
    
    def create_feathered_mask(self, mask, expansion, feathering):
        """Create mask with expansion and feathering using OpenCV (Optimized)"""
        if expansion == 0 and feathering == 0:
            return mask
        
        # Ensure mask is single channel numpy array
        if mask.ndim == 3:
            mask = mask[:, :, 0] if mask.shape[2] > 1 else mask[:, :, 0]
        
        # Convert to 8-bit uint (0-255 range)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Expand mask - using OpenCV morphological operations (much faster than PIL)
        if expansion > 0:
            # Create elliptical structuring element, size adjusted by expansion
            kernel_size = max(3, expansion * 2 + 1)  # Ensure odd
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Use dilate for expansion (single operation)
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
        
        # Feather mask - using OpenCV Gaussian blur (much faster than PIL)
        if feathering > 0:
            # Calculate Gaussian kernel size (must be positive odd)
            kernel_size = max(3, feathering * 2 + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd
                
            # Apply Gaussian blur
            mask_uint8 = cv2.GaussianBlur(mask_uint8, (kernel_size, kernel_size), feathering)
        
        # Convert back to float and normalize
        feathered_mask = mask_uint8.astype(np.float32) / 255.0
        
        return feathered_mask
    
    def restore_mask_to_original_position(self, cropped_mask, seam):
        """Restore cropped mask to same size and position as unscaled crop mask"""
        # Get seam data
        original_height, original_width = seam["original_size"]
        crop_top, crop_bottom, crop_left, crop_right = seam["crop_region"]
        target_height, target_width = seam["target_size"]
        output_size_mode = seam.get("output_size_mode", "Original Pixel")
        padding_info = seam.get("padding_info", None)
        
        # Calculate original crop region dimensions
        original_crop_height = crop_bottom - crop_top
        original_crop_width = crop_right - crop_left
        
        # Convert cropped mask to numpy
        cropped_mask_np = cropped_mask[0].cpu().numpy()
        
        # Handle potential 3D shape (C, H, W) or (H, W, C)
        if cropped_mask_np.ndim == 3:
            # Handle (C, H, W) where C=1
            if cropped_mask_np.shape[0] == 1:
                cropped_mask_np = cropped_mask_np[0]
            # Handle (H, W, C) where C=1
            elif cropped_mask_np.shape[-1] == 1:
                cropped_mask_np = cropped_mask_np[..., 0]

        process_height, process_width = cropped_mask_np.shape[:2]
        
        # Check if dimensions match
        if process_height != target_height or process_width != target_width:
            # If not match, resize to target size first
            pil_cropped_mask = Image.fromarray((cropped_mask_np * 255).astype(np.uint8))
            pil_cropped_mask = pil_cropped_mask.resize((target_width, target_height), Image.LANCZOS)
            cropped_mask_np = np.array(pil_cropped_mask).astype(np.float32) / 255.0
        
        # For Custom Dimensions mode, need to restore to expanded size first
        if output_size_mode == "Custom Dimensions":
            # If there is padding info, remove padding first
            if padding_info is not None:
                left_pad, top_pad, scaled_w, scaled_h, target_w, target_h = padding_info
                # Extract valid region from processed image (remove padding)
                valid_region = cropped_mask_np[top_pad:top_pad+scaled_h, left_pad:left_pad+scaled_w]
                cropped_mask_np = valid_region
                process_height, process_width = cropped_mask_np.shape[:2]
            
            # Resize to original crop dimensions
            pil_cropped_mask = Image.fromarray((cropped_mask_np * 255).astype(np.uint8))
            restored_pil = pil_cropped_mask.resize((original_crop_width, original_crop_height), Image.LANCZOS)
            restored_np = np.array(restored_pil).astype(np.float32) / 255.0
        else:
            # Other modes resize directly to original crop dimensions
            pil_cropped_mask = Image.fromarray((cropped_mask_np * 255).astype(np.uint8))
            restored_pil = pil_cropped_mask.resize((original_crop_width, original_crop_height), Image.LANCZOS)
            restored_np = np.array(restored_pil).astype(np.float32) / 255.0
        
        return restored_np
    
    def check_if_mask_is_black(self, mask):
        """Check if mask is all black (almost no white pixels)"""
        if mask is None:
            return True
        # Calculate proportion of non-zero pixels
        non_zero_ratio = np.sum(mask > 0.1) / mask.size
        return non_zero_ratio < 0.001  # If non-zero ratio < 0.1%, consider it all black
    
    def restore_image(self, seam, cropped_image, mask_expansion_percentage, mask_feathering_percentage, cropped_mask=None, background_image=None):
        # Add RGB conversion
        cropped_image_rgb = self.image_to_rgb(cropped_image)[0]
        if background_image is not None:
            background_image_rgb = self.image_to_rgb(background_image)[0]
        else:
            background_image_rgb = None
        
        # Get seam data
        original_height, original_width = seam["original_size"]
        crop_top, crop_bottom, crop_left, crop_right = seam["crop_region"]
        target_height, target_width = seam["target_size"]
        scale_ratio_width = seam["scale_ratio_width"]
        scale_ratio_height = seam["scale_ratio_height"]
        final_scale_width = seam.get("final_scale_width", 1.0)
        final_scale_height = seam.get("final_scale_height", 1.0)
        output_size_mode = seam.get("output_size_mode", "Original Pixel")
        original_image = seam["original_image"]
        mask_fill = seam.get("mask_fill", False)
        mask_white_region_width = seam.get("mask_white_region_width", 0)
        mask_white_region_height = seam.get("mask_white_region_height", 0)
        padding_info = seam.get("padding_info", None)
        unscaled_crop_mask = seam.get("unscaled_crop_mask", None)
        
        # Use filled mask (if exists and mask fill enabled), otherwise use original mask
        if mask_fill and "filled_mask" in seam:
            original_mask = seam["filled_mask"]
        else:
            original_mask = seam["original_mask"]
        
        # Convert percentage to pixel values
        mask_expansion = self.calculate_percentage_pixel_value(mask_expansion_percentage, mask_white_region_width, mask_white_region_height)
        mask_feathering = self.calculate_percentage_pixel_value(mask_feathering_percentage, mask_white_region_width, mask_white_region_height)
        
        # Get background image
        if background_image_rgb is not None:
            # Check if background image dimensions match
            background_height, background_width = background_image_rgb.shape[1:3]
            if background_height != original_height or background_width != original_width:
                raise ValueError(f"Background image dimensions ({background_width}x{background_height}) do not match original image dimensions ({original_width}x{original_height})")
            background = background_image_rgb[0].cpu().numpy()
        else:
            # Use original image from seam as background, also needs RGB conversion
            original_image_rgb = self.image_to_rgb(original_image)[0]
            background = original_image_rgb[0].cpu().numpy()
        
        # Convert cropped image to numpy
        cropped_np = cropped_image_rgb[0].cpu().numpy()
        process_height, process_width = cropped_np.shape[:2]
        
        force_match_size = True  # Hardcoded to True
        if force_match_size and (process_height != target_height or process_width != target_width):
            pil_cropped = Image.fromarray((cropped_np * 255).astype(np.uint8))
            pil_cropped = pil_cropped.resize((target_width, target_height), Image.LANCZOS)
            cropped_np = np.array(pil_cropped).astype(np.float32) / 255.0
            process_height, process_width = cropped_np.shape[:2]

        
        # Check if dimensions match (if still not match after force match, use original logic)
        if process_height != target_height or process_width != target_width:
            # If not match, resize to target size first
            pil_cropped = Image.fromarray((cropped_np * 255).astype(np.uint8))
            pil_cropped = pil_cropped.resize((target_width, target_height), Image.LANCZOS)
            cropped_np = np.array(pil_cropped).astype(np.float32) / 255.0
        
        # Calculate original crop region dimensions
        original_crop_height = crop_bottom - crop_top
        original_crop_width = crop_right - crop_left
        
        # For Custom Dimensions mode, need to restore to expanded size first
        if output_size_mode == "Custom Dimensions":
            # If there is padding info, remove padding first
            if padding_info is not None:
                left_pad, top_pad, scaled_w, scaled_h, target_w, target_h = padding_info
                # Extract valid region from processed image (remove padding)
                valid_region = cropped_np[top_pad:top_pad+scaled_h, left_pad:left_pad+scaled_w]
                cropped_np = valid_region
                process_height, process_width = cropped_np.shape[:2]
            
            # Resize to original crop dimensions
            pil_cropped = Image.fromarray((cropped_np * 255).astype(np.uint8))
            restored_pil = pil_cropped.resize((original_crop_width, original_crop_height), Image.LANCZOS)
            restored_np = np.array(restored_pil).astype(np.float32) / 255.0
        else:
            # Other modes resize directly to original crop dimensions
            pil_cropped = Image.fromarray((cropped_np * 255).astype(np.uint8))
            restored_pil = pil_cropped.resize((original_crop_width, original_crop_height), Image.LANCZOS)
            restored_np = np.array(restored_pil).astype(np.float32) / 255.0
        
        # Key modification: Prioritize using passed cropped mask if it exists and is not all black
        if cropped_mask is not None and not self.check_if_mask_is_black(cropped_mask[0].cpu().numpy()):
            # Use passed cropped mask, restore to original position first
            restored_mask_np = self.restore_mask_to_original_position(cropped_mask, seam)
        else:
            # Use unscaled cropped mask to calculate mask position
            if unscaled_crop_mask is not None:
                # Resize unscaled cropped mask to original crop dimensions
                unscaled_mask_np = np.array(unscaled_crop_mask).astype(np.float32) / 255.0
                pil_unscaled_mask = Image.fromarray((unscaled_mask_np * 255).astype(np.uint8))
                restored_mask_pil = pil_unscaled_mask.resize((original_crop_width, original_crop_height), Image.LANCZOS)
                restored_mask_np = np.array(restored_mask_pil).astype(np.float32) / 255.0
            else:
                # If no unscaled crop mask saved, use original method
                # Get corresponding region of original mask
                original_mask_np = original_mask[0].cpu().numpy()
                
                # Calculate valid region of mask in original image
                mask_top = max(0, crop_top)
                mask_bottom = min(original_height, crop_bottom)
                mask_left = max(0, crop_left)
                mask_right = min(original_width, crop_right)
                
                mask_region = original_mask_np[mask_top:mask_bottom, mask_left:mask_right]
                
                # Resize mask to restored dimensions
                if mask_region.size > 0:
                    pil_mask = Image.fromarray((mask_region * 255).astype(np.uint8))
                    restored_mask_pil = pil_mask.resize((original_crop_width, original_crop_height), Image.LANCZOS)
                    restored_mask_np = np.array(restored_mask_pil).astype(np.float32) / 255.0
                else:
                    restored_mask_np = np.zeros((original_crop_height, original_crop_width), dtype=np.float32)
        
        # Apply mask expansion and feathering (using optimized OpenCV method)
        feathered_mask = self.create_feathered_mask(restored_mask_np, mask_expansion, mask_feathering)
        
        # Ensure mask is 3-channel
        if restored_np.ndim == 3:
            mask_3d = np.stack([feathered_mask, feathered_mask, feathered_mask], axis=-1)
        else:
            mask_3d = feathered_mask
        
        # Blend cropped image into background
        result = background.copy()
        
        # Calculate paste position (only paste parts within original image)
        paste_x = max(0, crop_left)
        paste_y = max(0, crop_top)
        
        # Calculate corresponding region in restored image
        source_x_offset = max(0, -crop_left)
        source_y_offset = max(0, -crop_top)
        source_width = min(original_crop_width - source_x_offset, original_width - paste_x)
        source_height = min(original_crop_height - source_y_offset, original_height - paste_y)
        
        if source_width > 0 and source_height > 0:
            # Extract region to paste
            paste_region = result[paste_y:paste_y+source_height, paste_x:paste_x+source_width]
            
            # Extract corresponding restored image region
            restored_region = restored_np[source_y_offset:source_y_offset+source_height, source_x_offset:source_x_offset+source_width]
            
            # Extract corresponding mask region
            mask_region = mask_3d[source_y_offset:source_y_offset+source_height, source_x_offset:source_x_offset+source_width]
            
            # Blend using mask
            blended_region = paste_region * (1 - mask_region) + restored_region * mask_region
            result[paste_y:paste_y+source_height, paste_x:paste_x+source_width] = blended_region
        
        # Convert back to tensor
        result_tensor = torch.from_numpy(result)[None, ...]
        
        return (result_tensor,)

class GH_MaskCropV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "遮罩": ("MASK",),
                "遮罩填充": ("BOOLEAN", {"default": False}),
                "扩展系数上": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 3.0, "step": 0.1}),
                "扩展系数下": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 3.0, "step": 0.1}),
                "扩展系数左": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 3.0, "step": 0.1}),
                "扩展系数右": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 3.0, "step": 0.1}),
                "输出尺寸": (["原像素", "原像素1：1", "自定义宽高"], {"default": "原像素"}),
                "自定义宽": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "自定义高": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "倍数取整": ("INT", {"default": 8, "min": 0, "max": 256}),
            }
        }
    
    RETURN_TYPES = ("SEAM", "IMAGE", "MASK")
    RETURN_NAMES = ("接缝", "裁剪图像", "裁剪遮罩")
    FUNCTION = "裁剪图像"
    CATEGORY = "孤海工具箱"
    
    def 填充遮罩孔洞(self, 遮罩_np):
        """填充遮罩内部的孔洞"""
        from scipy import ndimage
        
        # 二值化遮罩
        二值遮罩 = 遮罩_np > 0.1
        
        # 使用形态学操作填充孔洞
        填充遮罩 = ndimage.binary_fill_holes(二值遮罩)
        
        # 转换回原始范围
        填充遮罩 = 填充遮罩.astype(np.float32)
        
        return 填充遮罩
    
    def 计算倍数取整(self, 值, 倍数):
        """修改为向下取整"""
        if 倍数 <= 1:
            return 值
        return math.floor(值 / 倍数) * 倍数
    
    def 获取颜色众数(self, 图像, 区域, 方向, 采样宽度=10):
        """获取指定区域边缘颜色的众数（出现频率最高的颜色）"""
        if 图像.mode != "RGB":
            图像 = 图像.convert("RGB")
        
        宽, 高 = 图像.size
        左, 上, 右, 下 = 区域
        
        # 定义采样区域
        if 方向 == "左" and 左 < 0:
            # 取最左侧的一列像素进行采样
            采样左 = 0
            采样右 = min(采样宽度, 宽)
            采样上 = max(0, 上)
            采样下 = min(高, 下)
        elif 方向 == "右" and 右 > 宽:
            # 取最右侧的一列像素进行采样
            采样右 = 宽
            采样左 = max(0, 宽 - 采样宽度)
            采样上 = max(0, 上)
            采样下 = min(高, 下)
        elif 方向 == "上" and 上 < 0:
            # 取最上方的一行像素进行采样
            采样上 = 0
            采样下 = min(采样宽度, 高)
            采样左 = max(0, 左)
            采样右 = min(宽, 右)
        elif 方向 == "下" and 下 > 高:
            # 取最下方的一行像素进行采样
            采样下 = 高
            采样上 = max(0, 高 - 采样宽度)
            采样左 = max(0, 左)
            采样右 = min(宽, 右)
        else:
            # 不需要填充或方向无效，返回黑色
            return (0, 0, 0)
        
        # 提取采样区域的像素
        if 采样右 > 采样左 and 采样下 > 采样上:
            采样区域 = 图像.crop((采样左, 采样上, 采样右, 采样下))
            像素数据 = list(采样区域.getdata())
            
            if 像素数据:
                # 对颜色进行量化（将相近颜色归为一类）
                量化像素 = []
                for r, g, b in 像素数据:
                    # 将颜色量化为16级，减少颜色种类
                    量化_r = (r // 16) * 16
                    量化_g = (g // 16) * 16
                    量化_b = (b // 16) * 16
                    量化像素.append((量化_r, 量化_g, 量化_b))
                
                # 计算众数颜色
                颜色计数 = Counter(量化像素)
                众数颜色 = 颜色计数.most_common(1)[0][0]
                return 众数颜色
        
        return (0, 0, 0)
    
    def 获取边缘像素填充图像(self, 图像, 目标区域, 原始宽, 原始高, is_mask=False):
        """使用众数颜色进行填充，每个需要填充的方向都使用该方向的边缘众数颜色填充"""
        左, 上, 右, 下 = 目标区域
        
        # 计算实际需要的尺寸
        目标宽 = 右 - 左
        目标高 = 下 - 上
        
        # 创建目标图像
        if is_mask:
            # 对于遮罩，超出部分始终用黑色填充
            目标图像 = Image.new("L", (目标宽, 目标高), 0)
        else:
            # 对于图像，使用原图模式
            目标图像 = Image.new(图像.mode, (目标宽, 目标高))
            
            # 预计算各方向的众数颜色
            左众数 = self.获取颜色众数(图像, (左, 上, 右, 下), "左")
            右众数 = self.获取颜色众数(图像, (左, 上, 右, 下), "右")
            上众数 = self.获取颜色众数(图像, (左, 上, 右, 下), "上")
            下众数 = self.获取颜色众数(图像, (左, 上, 右, 下), "下")
        
        # 计算在原始图像内的有效区域
        有效左 = max(0, 左)
        有效上 = max(0, 上)
        有效右 = min(原始宽, 右)
        有效下 = min(原始高, 下)
        
        # 计算在目标图像中的对应位置
        目标左偏移 = 有效左 - 左
        目标上偏移 = 有效上 - 上
        
        # 如果有有效区域，从原图复制
        if 有效右 > 有效左 and 有效下 > 有效上:
            有效区域 = 图像.crop((有效左, 有效上, 有效右, 有效下))
            目标图像.paste(有效区域, (目标左偏移, 目标上偏移))
        
        # 对于图像，使用众数颜色填充超出边界的部分
        if not is_mask:
            # 判断需要填充的方向（每个方向独立判断，不再限制只填充一个方向）
            需要填充左 = 左 < 0
            需要填充右 = 右 > 原始宽
            需要填充上 = 上 < 0
            需要填充下 = 下 > 原始高
            
            # 左侧填充
            if 需要填充左:
                填充宽度 = -左
                for x in range(填充宽度):
                    for y in range(目标高):
                        目标图像.putpixel((x, y), 左众数)
            
            # 右侧填充
            if 需要填充右:
                填充宽度 = 右 - 原始宽
                起始x = 目标宽 - 填充宽度
                for x in range(起始x, 目标宽):
                    for y in range(目标高):
                        目标图像.putpixel((x, y), 右众数)
            
            # 上方填充
            if 需要填充上:
                填充高度 = -上
                for y in range(填充高度):
                    for x in range(目标宽):
                        目标图像.putpixel((x, y), 上众数)
            
            # 下方填充
            if 需要填充下:
                填充高度 = 下 - 原始高
                起始y = 目标高 - 填充高度
                for y in range(起始y, 目标高):
                    for x in range(目标宽):
                        目标图像.putpixel((x, y), 下众数)
        
        return 目标图像
    
    def 裁剪图像(self, 图像, 遮罩, 遮罩填充, 扩展系数上, 扩展系数下, 扩展系数左, 扩展系数右, 
                  输出尺寸, 自定义宽, 自定义高, 倍数取整):
        # 确保输入是单张图像
        if 图像.shape[0] > 1:
            图像 = 图像[0:1]
        if 遮罩.shape[0] > 1:
            遮罩 = 遮罩[0:1]
            
        # 转换tensor为numpy数组
        图像_np = 图像[0].cpu().numpy()
        遮罩_np = 遮罩[0].cpu().numpy()
        
        # 保存填充后的遮罩到接缝数据
        填充后遮罩_np = 遮罩_np.copy()
        
        # 根据遮罩填充开关处理遮罩
        if 遮罩填充:
            填充后遮罩_np = self.填充遮罩孔洞(遮罩_np)
        
        # 获取原像素
        原始高, 原始宽 = 图像_np.shape[:2]
        
        # 找到遮罩的非零区域边界（使用填充后的遮罩）
        非零区域 = np.where(填充后遮罩_np > 0.1)  # 使用阈值避免噪点
        if len(非零区域[0]) == 0:
            # 如果没有遮罩区域，使用整个图像
            最小y, 最大y, 最小x, 最大x = 0, 原始高, 0, 原始宽
            遮罩宽 = 原始宽
            遮罩高 = 原始高
        else:
            最小y, 最大y = np.min(非零区域[0]), np.max(非零区域[0])
            最小x, 最大x = np.min(非零区域[1]), np.max(非零区域[1])
            遮罩宽 = 最大x - 最小x
            遮罩高 = 最大y - 最小y
        
        # 计算遮罩最短边
        遮罩最短边 = min(遮罩宽, 遮罩高)
        
        # 计算四个方向的扩展量（分别乘以遮罩最短边像素值*(扩展系数-1)）
        扩展量上 = int(遮罩最短边 * (扩展系数上 - 1.0))
        扩展量下 = int(遮罩最短边 * (扩展系数下 - 1.0))
        扩展量左 = int(遮罩最短边 * (扩展系数左 - 1.0))
        扩展量右 = int(遮罩最短边 * (扩展系数右 - 1.0))
        
        # 限制扩展量不超过原图边界
        扩展量上 = min(扩展量上, 最小y)  # 向上扩展不能超过图像顶部
        扩展量下 = min(扩展量下, 原始高 - 最大y - 1)  # 向下扩展不能超过图像底部
        扩展量左 = min(扩展量左, 最小x)  # 向左扩展不能超过图像左侧
        扩展量右 = min(扩展量右, 原始宽 - 最大x - 1)  # 向右扩展不能超过图像右侧
        
        # 计算基础裁剪区域（遮罩区域 + 各方向扩展量）
        基础顶部 = 最小y - 扩展量上
        基础底部 = 最大y + 扩展量下
        基础左侧 = 最小x - 扩展量左
        基础右侧 = 最大x + 扩展量右
        
        # 根据输出尺寸选项确定最终裁剪区域
        if 输出尺寸 == "原像素":
            # 原像素：遮罩+扩展量，然后倍数取整
            裁剪宽 = 基础右侧 - 基础左侧
            裁剪高 = 基础底部 - 基础顶部
            
            if 倍数取整 > 1:
                裁剪宽 = self.计算倍数取整(裁剪宽, 倍数取整)
                裁剪高 = self.计算倍数取整(裁剪高, 倍数取整)
            
            # 保持中心点不变，调整尺寸
            中心x = (基础左侧 + 基础右侧) // 2
            中心y = (基础顶部 + 基础底部) // 2
            
            最终左侧 = 中心x - 裁剪宽 // 2
            最终右侧 = 最终左侧 + 裁剪宽
            最终顶部 = 中心y - 裁剪高 // 2
            最终底部 = 最终顶部 + 裁剪高
            
        elif 输出尺寸 == "原像素1：1":
            # 原像素1：1：取最小外接正方形，然后倍数取整
            基础宽 = 基础右侧 - 基础左侧
            基础高 = 基础底部 - 基础顶部
            目标尺寸 = max(基础宽, 基础高)
            
            if 倍数取整 > 1:
                目标尺寸 = self.计算倍数取整(目标尺寸, 倍数取整)
            
            # 保持中心点不变，调整尺寸
            中心x = (基础左侧 + 基础右侧) // 2
            中心y = (基础顶部 + 基础底部) // 2
            
            最终左侧 = 中心x - 目标尺寸 // 2
            最终右侧 = 最终左侧 + 目标尺寸
            最终顶部 = 中心y - 目标尺寸 // 2
            最终底部 = 最终顶部 + 目标尺寸
            
        else:  # 自定义宽高
            # 先对用户输入进行倍数取整
            目标宽 = 自定义宽
            目标高 = 自定义高
            if 倍数取整 > 1:
                目标宽 = self.计算倍数取整(目标宽, 倍数取整)
                目标高 = self.计算倍数取整(目标高, 倍数取整)
            
            # 计算基础宽高比和目标宽高比
            基础宽 = 基础右侧 - 基础左侧
            基础高 = 基础底部 - 基础顶部
            基础宽高比 = 基础宽 / 基础高
            目标宽高比 = 目标宽 / 目标高
            
            # 确定扩展方向
            if 基础宽高比 < 目标宽高比:
                # 需要扩展宽度
                扩展后宽 = int(基础高 * 目标宽高比)
                扩展后高 = 基础高
            else:
                # 需要扩展高度
                扩展后宽 = 基础宽
                扩展后高 = int(基础宽 / 目标宽高比)
            
            # 保持中心点不变，调整尺寸
            中心x = (基础左侧 + 基础右侧) // 2
            中心y = (基础顶部 + 基础底部) // 2
            
            最终左侧 = 中心x - 扩展后宽 // 2
            最终右侧 = 最终左侧 + 扩展后宽
            最终顶部 = 中心y - 扩展后高 // 2
            最终底部 = 最终顶部 + 扩展后高
        
        # 转换为整数坐标
        最终左侧 = int(最终左侧)
        最终右侧 = int(最终右侧)
        最终顶部 = int(最终顶部)
        最终底部 = int(最终底部)
        
        # 计算实际裁剪尺寸
        裁剪宽 = 最终右侧 - 最终左侧
        裁剪高 = 最终底部 - 最终顶部
        
        # 转换为PIL图像进行处理
        pil图像 = Image.fromarray((图像_np * 255).astype(np.uint8))
        pil遮罩 = Image.fromarray((填充后遮罩_np * 255).astype(np.uint8))
        
        # 获取裁剪区域（使用新的填充方法）
        裁剪图像 = self.获取边缘像素填充图像(pil图像, (最终左侧, 最终顶部, 最终右侧, 最终底部), 原始宽, 原始高, is_mask=False)
        裁剪遮罩 = self.获取边缘像素填充图像(pil遮罩, (最终左侧, 最终顶部, 最终右侧, 最终底部), 原始宽, 原始高, is_mask=True)
        
        # 保存未缩放的裁剪图像和遮罩，用于恢复时计算位置
        未缩放裁剪图像 = 裁剪图像.copy()
        未缩放裁剪遮罩 = 裁剪遮罩.copy()
        
        # 对于自定义宽高模式，需要等比例缩放到目标尺寸
        if 输出尺寸 == "自定义宽高":
            当前宽, 当前高 = 裁剪图像.size
            缩放比例宽 = 目标宽 / 当前宽
            缩放比例高 = 目标高 / 当前高
            缩放比例 = min(缩放比例宽, 缩放比例高)
            
            缩放宽 = int(当前宽 * 缩放比例)
            缩放高 = int(当前高 * 缩放比例)
            
            裁剪图像 = 裁剪图像.resize((缩放宽, 缩放高), Image.LANCZOS)
            裁剪遮罩 = 裁剪遮罩.resize((缩放宽, 缩放高), Image.LANCZOS)
            
            # 如果需要，填充到目标尺寸
            if 缩放宽 != 目标宽 or 缩放高 != 目标高:
                # 创建目标尺寸图像，用黑色填充
                最终图像 = Image.new(裁剪图像.mode, (目标宽, 目标高), 0)
                最终遮罩 = Image.new(裁剪遮罩.mode, (目标宽, 目标高), 0)
                
                # 计算粘贴位置（居中）
                左偏移 = (目标宽 - 缩放宽) // 2
                上偏移 = (目标高 - 缩放高) // 2
                
                最终图像.paste(裁剪图像, (左偏移, 上偏移))
                最终遮罩.paste(裁剪遮罩, (左偏移, 上偏移))
                
                裁剪图像 = 最终图像
                裁剪遮罩 = 最终遮罩
                
                # 保存填充信息
                填充信息 = (左偏移, 上偏移, 缩放宽, 缩放高, 目标宽, 目标高)
            else:
                填充信息 = None
        else:
            目标宽, 目标高 = 裁剪图像.size
            填充信息 = None
        
        # 转换回numpy数组
        最终图像_np = np.array(裁剪图像).astype(np.float32) / 255.0
        最终遮罩_np = np.array(裁剪遮罩).astype(np.float32) / 255.0
        
        # 转换为tensor
        最终图像_tensor = torch.from_numpy(最终图像_np)[None, ...]
        最终遮罩_tensor = torch.from_numpy(最终遮罩_np)[None, ...]
        
        # 计算缩放比例（用于恢复）
        if 输出尺寸 == "自定义宽高":
            # 自定义宽高模式下，有两次缩放：从裁剪区域到扩展区域，再到目标尺寸
            原始裁剪宽 = 最终右侧 - 最终左侧
            原始裁剪高 = 最终底部 - 最终顶部
            扩展后宽 = 未缩放裁剪图像.size[0]
            扩展后高 = 未缩放裁剪图像.size[1]
            
            缩放比例宽 = 扩展后宽 / 原始裁剪宽
            缩放比例高 = 扩展后高 / 原始裁剪高
            最终缩放宽 = 目标宽 / 扩展后宽 if 输出尺寸 == "自定义宽高" else 1.0
            最终缩放高 = 目标高 / 扩展后高 if 输出尺寸 == "自定义宽高" else 1.0
        else:
            # 其他模式下，只有一次缩放：从裁剪区域到目标尺寸
            原始裁剪宽 = 最终右侧 - 最终左侧
            原始裁剪高 = 最终底部 - 最终顶部
            缩放比例宽 = 目标宽 / 原始裁剪宽
            缩放比例高 = 目标高 / 原始裁剪高
            最终缩放宽 = 1.0
            最终缩放高 = 1.0
        
        # 创建接缝数据（包含所有必要信息）
        接缝数据 = {
            "原始图像": 图像,  # 原始图像tensor
            "原始遮罩": 遮罩,    # 原始遮罩tensor
            "填充后遮罩": torch.from_numpy(填充后遮罩_np)[None, ...],  
            "原像素": (原始高, 原始宽),
            "裁剪区域": (最终顶部, 最终底部, 最终左侧, 最终右侧),
            "目标尺寸": (目标高, 目标宽),
            "缩放比例宽": 缩放比例宽,
            "缩放比例高": 缩放比例高,
            "最终缩放宽": 最终缩放宽,
            "最终缩放高": 最终缩放高,
            "输出尺寸模式": 输出尺寸,
            "填充信息": 填充信息,  # 保存填充信息用于恢复
            "遮罩填充": 遮罩填充, 
            "遮罩白色区域宽": 遮罩宽, 
            "遮罩白色区域高": 遮罩高,
            "未缩放裁剪遮罩": 未缩放裁剪遮罩  # 保存未缩放的裁剪遮罩用于恢复
        }
        
        return (接缝数据, 最终图像_tensor, 最终遮罩_tensor)


class GH_CropRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "接缝": ("SEAM",),
                "裁剪图像": ("IMAGE",),
                "遮罩扩展百分比": ("INT", {"default": 5, "min": 0, "max": 30, "step": 1}),
                "遮罩羽化百分比": ("INT", {"default": 5, "min": 0, "max": 30, "step": 1}),
            },
            "optional": {
                "裁剪遮罩": ("MASK",),
                "背景图像": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "恢复图像"
    CATEGORY = "孤海工具箱"
    
    def tensor2pil(self, image):
        """将tensor转换为PIL图像"""
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
    def pil2tensor(self, image):
        """将PIL图像转换为tensor"""
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
    
    def image_to_rgb(self, images):
        """将图像转换为RGB格式"""
        if len(images) > 1:
            tensors = []
            for image in images:
                pil_image = self.tensor2pil(image)
                pil_image = pil_image.convert('RGB')
                tensors.append(self.pil2tensor(pil_image))
            tensors = torch.cat(tensors, dim=0)
            return (tensors, )
        else:
            pil_image = self.tensor2pil(images)
            pil_image = pil_image.convert('RGB')
            return (self.pil2tensor(pil_image), )
    
    def 计算百分比像素值(self, 百分比, 遮罩宽, 遮罩高):
        """将百分比转换为像素值，基于遮罩白色区域宽高的平均值"""
        if 百分比 == 0:
            return 0
        平均值 = (遮罩宽 + 遮罩高) / 2
        像素值 = int(平均值 * 百分比 / 100)
        return max(1, 像素值)  # 确保至少为1像素，避免为0
    
    def 创建羽化遮罩(self, 遮罩, 扩展, 羽化):
        """使用OpenCV创建带有扩展和羽化效果的遮罩（性能优化版）"""
        if 扩展 == 0 and 羽化 == 0:
            return 遮罩
        
        # 确保遮罩是单通道的numpy数组
        if 遮罩.ndim == 3:
            遮罩 = 遮罩[:, :, 0] if 遮罩.shape[2] > 1 else 遮罩[:, :, 0]
        
        # 转换为8位无符号整数（0-255范围）
        遮罩_uint8 = (遮罩 * 255).astype(np.uint8)
        
        # 扩展遮罩 - 使用OpenCV的形态学操作（比PIL快得多）
        if 扩展 > 0:
            # 创建椭圆形的结构元素，大小根据扩展值调整
            核大小 = max(3, 扩展 * 2 + 1)  # 确保是奇数
            核 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (核大小, 核大小))
            
            # 使用膨胀操作进行扩展（单次操作完成，而不是循环多次）
            遮罩_uint8 = cv2.dilate(遮罩_uint8, 核, iterations=1)
        
        # 羽化遮罩 - 使用OpenCV的高斯模糊（比PIL快得多）
        if 羽化 > 0:
            # 计算高斯核大小（必须是正奇数）
            核大小 = max(3, 羽化 * 2 + 1)
            if 核大小 % 2 == 0:
                核大小 += 1  # 确保是奇数
                
            # 应用高斯模糊
            遮罩_uint8 = cv2.GaussianBlur(遮罩_uint8, (核大小, 核大小), 羽化)
        
        # 转换回浮点数并归一化
        羽化遮罩 = 遮罩_uint8.astype(np.float32) / 255.0
        
        return 羽化遮罩
    
    def 恢复遮罩到原始位置(self, 裁剪遮罩, 接缝):
        """将裁剪遮罩恢复到与未缩放裁剪遮罩相同的大小和位置"""
        # 获取接缝数据
        原始高, 原始宽 = 接缝["原像素"]
        裁剪顶部, 裁剪底部, 裁剪左侧, 裁剪右侧 = 接缝["裁剪区域"]
        目标高, 目标宽 = 接缝["目标尺寸"]
        输出尺寸模式 = 接缝.get("输出尺寸模式", "原像素")
        填充信息 = 接缝.get("填充信息", None)
        
        # 计算原始裁剪区域的尺寸
        原始裁剪高 = 裁剪底部 - 裁剪顶部
        原始裁剪宽 = 裁剪右侧 - 裁剪左侧
        
        # 将裁剪遮罩转换为numpy
        裁剪遮罩_np = 裁剪遮罩[0].cpu().numpy()
        处理高, 处理宽 = 裁剪遮罩_np.shape[:2]
        
        # 检查尺寸是否匹配
        if 处理高 != 目标高 or 处理宽 != 目标宽:
            # 如果不匹配，先缩放到目标尺寸
            pil裁剪遮罩 = Image.fromarray((裁剪遮罩_np * 255).astype(np.uint8))
            pil裁剪遮罩 = pil裁剪遮罩.resize((目标宽, 目标高), Image.LANCZOS)
            裁剪遮罩_np = np.array(pil裁剪遮罩).astype(np.float32) / 255.0
        
        # 对于自定义宽高模式，需要先恢复到扩展后的尺寸
        if 输出尺寸模式 == "自定义宽高":
            # 如果有填充信息，先去除填充
            if 填充信息 is not None:
                左填充, 上填充, 缩放宽, 缩放高, 目标宽, 目标高 = 填充信息
                # 从处理后的图像中提取有效区域（去除填充）
                有效区域 = 裁剪遮罩_np[上填充:上填充+缩放高, 左填充:左填充+缩放宽]
                裁剪遮罩_np = 有效区域
                处理高, 处理宽 = 裁剪遮罩_np.shape[:2]
            
            # 缩放到原始裁剪尺寸
            pil裁剪遮罩 = Image.fromarray((裁剪遮罩_np * 255).astype(np.uint8))
            恢复后_pil = pil裁剪遮罩.resize((原始裁剪宽, 原始裁剪高), Image.LANCZOS)
            恢复后_np = np.array(恢复后_pil).astype(np.float32) / 255.0
        else:
            # 其他模式直接缩放到原始裁剪尺寸
            pil裁剪遮罩 = Image.fromarray((裁剪遮罩_np * 255).astype(np.uint8))
            恢复后_pil = pil裁剪遮罩.resize((原始裁剪宽, 原始裁剪高), Image.LANCZOS)
            恢复后_np = np.array(恢复后_pil).astype(np.float32) / 255.0
        
        return 恢复后_np
    
    def 检查遮罩是否全黑(self, 遮罩):
        """检查遮罩是否全黑（几乎没有白色像素）"""
        if 遮罩 is None:
            return True
        # 计算非零像素的比例
        非零像素比例 = np.sum(遮罩 > 0.1) / 遮罩.size
        return 非零像素比例 < 0.001  # 如果非零像素比例小于0.1%，认为是全黑
    
    def 恢复图像(self, 接缝, 裁剪图像, 遮罩扩展百分比, 遮罩羽化百分比, 裁剪遮罩=None, 背景图像=None):
        # 添加RGB转换功能
        裁剪图像_rgb = self.image_to_rgb(裁剪图像)[0]
        if 背景图像 is not None:
            背景图像_rgb = self.image_to_rgb(背景图像)[0]
        else:
            背景图像_rgb = None
        
        # 获取接缝数据
        原始高, 原始宽 = 接缝["原像素"]
        裁剪顶部, 裁剪底部, 裁剪左侧, 裁剪右侧 = 接缝["裁剪区域"]
        目标高, 目标宽 = 接缝["目标尺寸"]
        缩放比例宽 = 接缝["缩放比例宽"]
        缩放比例高 = 接缝["缩放比例高"]
        最终缩放宽 = 接缝.get("最终缩放宽", 1.0)
        最终缩放高 = 接缝.get("最终缩放高", 1.0)
        输出尺寸模式 = 接缝.get("输出尺寸模式", "原像素")
        原始图像 = 接缝["原始图像"]
        遮罩填充 = 接缝.get("遮罩填充", False)
        遮罩白色区域宽 = 接缝.get("遮罩白色区域宽", 0)
        遮罩白色区域高 = 接缝.get("遮罩白色区域高", 0)
        填充信息 = 接缝.get("填充信息", None)
        未缩放裁剪遮罩 = 接缝.get("未缩放裁剪遮罩", None)
        
        # 使用填充后的遮罩（如果存在且开启了遮罩填充），否则使用原始遮罩
        if 遮罩填充 and "填充后遮罩" in 接缝:
            原始遮罩 = 接缝["填充后遮罩"]
        else:
            原始遮罩 = 接缝["原始遮罩"]
        
        # 将百分比转换为像素值
        遮罩扩展 = self.计算百分比像素值(遮罩扩展百分比, 遮罩白色区域宽, 遮罩白色区域高)
        遮罩羽化 = self.计算百分比像素值(遮罩羽化百分比, 遮罩白色区域宽, 遮罩白色区域高)
        
        # 获取背景图像
        if 背景图像_rgb is not None:
            # 检查背景图像尺寸是否匹配
            背景高, 背景宽 = 背景图像_rgb.shape[1:3]
            if 背景高 != 原始高 or 背景宽 != 原始宽:
                raise ValueError(f"背景图像尺寸({背景宽}x{背景高})与原始图像尺寸({原始宽}x{原始高})不匹配")
            背景 = 背景图像_rgb[0].cpu().numpy()
        else:
            # 使用接缝中的原始图像作为背景，也需要转换为RGB
            原始图像_rgb = self.image_to_rgb(原始图像)[0]
            背景 = 原始图像_rgb[0].cpu().numpy()
        
        # 转换裁剪图像为numpy
        裁剪_np = 裁剪图像_rgb[0].cpu().numpy()
        处理高, 处理宽 = 裁剪_np.shape[:2]
        
        强制匹配尺寸 = True  # 硬编码为True
        if 强制匹配尺寸 and (处理高 != 目标高 or 处理宽 != 目标宽):
            pil裁剪 = Image.fromarray((裁剪_np * 255).astype(np.uint8))
            pil裁剪 = pil裁剪.resize((目标宽, 目标高), Image.LANCZOS)
            裁剪_np = np.array(pil裁剪).astype(np.float32) / 255.0
            处理高, 处理宽 = 裁剪_np.shape[:2]

        
        # 检查尺寸是否匹配（如果强制匹配后仍不匹配，则使用原始逻辑）
        if 处理高 != 目标高 or 处理宽 != 目标宽:
            # 如果不匹配，先缩放到目标尺寸
            pil裁剪 = Image.fromarray((裁剪_np * 255).astype(np.uint8))
            pil裁剪 = pil裁剪.resize((目标宽, 目标高), Image.LANCZOS)
            裁剪_np = np.array(pil裁剪).astype(np.float32) / 255.0
        
        # 计算原始裁剪区域的尺寸
        原始裁剪高 = 裁剪底部 - 裁剪顶部
        原始裁剪宽 = 裁剪右侧 - 裁剪左侧
        
        # 对于自定义宽高模式，需要先恢复到扩展后的尺寸
        if 输出尺寸模式 == "自定义宽高":
            # 如果有填充信息，先去除填充
            if 填充信息 is not None:
                左填充, 上填充, 缩放宽, 缩放高, 目标宽, 目标高 = 填充信息
                # 从处理后的图像中提取有效区域（去除填充）
                有效区域 = 裁剪_np[上填充:上填充+缩放高, 左填充:左填充+缩放宽]
                裁剪_np = 有效区域
                处理高, 处理宽 = 裁剪_np.shape[:2]
            
            # 缩放到原始裁剪尺寸
            pil裁剪 = Image.fromarray((裁剪_np * 255).astype(np.uint8))
            恢复后_pil = pil裁剪.resize((原始裁剪宽, 原始裁剪高), Image.LANCZOS)
            恢复后_np = np.array(恢复后_pil).astype(np.float32) / 255.0
        else:
            # 其他模式直接缩放到原始裁剪尺寸
            pil裁剪 = Image.fromarray((裁剪_np * 255).astype(np.uint8))
            恢复后_pil = pil裁剪.resize((原始裁剪宽, 原始裁剪高), Image.LANCZOS)
            恢复后_np = np.array(恢复后_pil).astype(np.float32) / 255.0
        
        # 关键修改：优先使用传入的裁剪遮罩，如果存在且不是全黑
        if 裁剪遮罩 is not None and not self.检查遮罩是否全黑(裁剪遮罩[0].cpu().numpy()):
            # 使用传入的裁剪遮罩，先恢复到原始位置
            恢复遮罩_np = self.恢复遮罩到原始位置(裁剪遮罩, 接缝)
        else:
            # 使用未缩放的裁剪遮罩来计算遮罩位置
            if 未缩放裁剪遮罩 is not None:
                # 将未缩放的裁剪遮罩缩放到原始裁剪尺寸
                未缩放遮罩_np = np.array(未缩放裁剪遮罩).astype(np.float32) / 255.0
                pil未缩放遮罩 = Image.fromarray((未缩放遮罩_np * 255).astype(np.uint8))
                恢复遮罩_pil = pil未缩放遮罩.resize((原始裁剪宽, 原始裁剪高), Image.LANCZOS)
                恢复遮罩_np = np.array(恢复遮罩_pil).astype(np.float32) / 255.0
            else:
                # 如果没有保存未缩放的裁剪遮罩，使用原始方法
                # 获取原始遮罩的对应区域
                原始遮罩_np = 原始遮罩[0].cpu().numpy()
                
                # 计算遮罩在原始图像中的有效区域
                遮罩顶部 = max(0, 裁剪顶部)
                遮罩底部 = min(原始高, 裁剪底部)
                遮罩左侧 = max(0, 裁剪左侧)
                遮罩右侧 = min(原始宽, 裁剪右侧)
                
                遮罩区域 = 原始遮罩_np[遮罩顶部:遮罩底部, 遮罩左侧:遮罩右侧]
                
                # 缩放遮罩到恢复后的尺寸
                if 遮罩区域.size > 0:
                    pil遮罩 = Image.fromarray((遮罩区域 * 255).astype(np.uint8))
                    恢复遮罩_pil = pil遮罩.resize((原始裁剪宽, 原始裁剪高), Image.LANCZOS)
                    恢复遮罩_np = np.array(恢复遮罩_pil).astype(np.float32) / 255.0
                else:
                    恢复遮罩_np = np.zeros((原始裁剪高, 原始裁剪宽), dtype=np.float32)
        
        # 应用遮罩扩展和羽化（使用优化后的OpenCV方法）
        羽化遮罩 = self.创建羽化遮罩(恢复遮罩_np, 遮罩扩展, 遮罩羽化)
        
        # 确保遮罩是3通道的
        if 恢复后_np.ndim == 3:
            遮罩3d = np.stack([羽化遮罩, 羽化遮罩, 羽化遮罩], axis=-1)
        else:
            遮罩3d = 羽化遮罩
        
        # 将裁剪图像融合到背景中
        结果 = 背景.copy()
        
        # 计算粘贴位置（只粘贴在原始图像内的部分）
        粘贴x = max(0, 裁剪左侧)
        粘贴y = max(0, 裁剪顶部)
        
        # 计算在恢复图像中对应的区域
        源x偏移 = max(0, -裁剪左侧)
        源y偏移 = max(0, -裁剪顶部)
        源宽 = min(原始裁剪宽 - 源x偏移, 原始宽 - 粘贴x)
        源高 = min(原始裁剪高 - 源y偏移, 原始高 - 粘贴y)
        
        if 源宽 > 0 and 源高 > 0:
            # 提取要粘贴的区域
            粘贴区域 = 结果[粘贴y:粘贴y+源高, 粘贴x:粘贴x+源宽]
            
            # 提取对应的恢复图像区域
            恢复区域 = 恢复后_np[源y偏移:源y偏移+源高, 源x偏移:源x偏移+源宽]
            
            # 提取对应的遮罩区域
            遮罩区域 = 遮罩3d[源y偏移:源y偏移+源高, 源x偏移:源x偏移+源宽]
            
            # 使用遮罩进行融合
            融合区域 = 粘贴区域 * (1 - 遮罩区域) + 恢复区域 * 遮罩区域
            结果[粘贴y:粘贴y+源高, 粘贴x:粘贴x+源宽] = 融合区域
        
        # 转换回tensor
        结果_tensor = torch.from_numpy(结果)[None, ...]
        
        return (结果_tensor,)
