from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
from typing import Sequence, Mapping, Any, Union
from .layerstyle.imagefunc import fit_resize_image


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        # Use alpha if available
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]
    # Convert RGB to grayscale
    return TF.rgb_to_grayscale(t.permute(0,3,1,2), num_output_channels=1)[:,0,:,:]

class EmptyImagePro:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "color": ("STRING", {"default": "0,0,0"}),
                              }}
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "generate"
    CATEGORY = "zdx/image"

    def generate(self, image, batch_size=1, color="0,0,0"):
        _, height, width, _ = image.shape
        try:
            color_list = [int(c.strip()) for c in color.split(',')]
            if len(color_list) < 3:
                print(f"Warning: Invalid color format '{color}'. Using black.")
                color_list = [0, 0, 0]
        except ValueError:
            print(f"Warning: Invalid color format '{color}'. Using black.")
            color_list = [0, 0, 0]
            
        r_val, g_val, b_val = color_list[0] / 255.0, color_list[1] / 255.0, color_list[2] / 255.0
        
        r = torch.full([batch_size, height, width, 1], r_val)
        g = torch.full([batch_size, height, width, 1], g_val)
        b = torch.full([batch_size, height, width, 1], b_val)
        empty_image = torch.cat((r, g, b), dim=-1)
        return (empty_image, tensor2mask(empty_image))

class MaskAreaReColor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                              "mask": ("MASK",),
                              "color": ("STRING", {"default": "0,0,0"}),
                              "invert_mask": ("BOOLEAN", {"default": True}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "zdx/image"

    def generate(self, image, mask, color="255,255,255", invert_mask=True):
        # Generate solid color image
        empty_image_pro = EmptyImagePro()
        solid_color_image, _ = empty_image_pro.generate(image, batch_size=image.shape[0], color=color)
        
        # Adjust mask shape to match image (B, H, W, 1)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(-1)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(-1)
            
        # Broadcast mask to match batch size if necessary
        if mask.shape[0] < image.shape[0]:
            mask = mask.repeat(image.shape[0], 1, 1, 1)
            
        if invert_mask:
            mask = 1.0 - mask
            
        # Composite: result = image * (1 - mask) + color * mask
        output = image * (1.0 - mask) + solid_color_image * mask
        return (output,)

class imageConcat:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "direction": (['right','down','left','up',],{"default": 'right'}),
            "match_image_size": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concat"
    CATEGORY = "zdx/Image"

    def concat(self, image1, image2, direction, match_image_size):
        if image1 is None:
            return (image2,)
        elif image2 is None:
            return (image1,)
        if match_image_size:
            # Convert tensor to PIL for proper aspect ratio resizing
            pil_image2 = tensor2pil(image2)
            if direction in ['right', 'left']:
                aspect_ratio = pil_image2.width / pil_image2.height
                new_height = image1.shape[1]
                new_width = int(aspect_ratio * new_height)
                pil_image2 = fit_resize_image(pil_image2, new_width, new_height, 'fill', Image.LANCZOS, '#000000')
            else:  # 'up' or 'down'
                aspect_ratio = pil_image2.height / pil_image2.width
                new_width = image1.shape[2]
                new_height = int(aspect_ratio * new_width)
                pil_image2 = fit_resize_image(pil_image2, new_width, new_height, 'fill', Image.LANCZOS, '#000000')
            image2 = pil2tensor(pil_image2)

        if direction == 'right':
            row = torch.cat((image1, image2), dim=2)
        elif direction == 'down':
            row = torch.cat((image1, image2), dim=1)
        elif direction == 'left':
            row = torch.cat((image2, image1), dim=2)
        elif direction == 'up':
            row = torch.cat((image2, image1), dim=1)
        return (row,)

class image_concat_mask_v1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",), "mask": ("MASK",),
                "invert": ("BOOLEAN", {"default": False}),
                "match_image_size": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_concat_mask"
    CATEGORY = "zdx/image"

    def image_concat_mask(self, image1, image2=None, mask=None, invert=True, match_image_size=False):
#-------concat norm image and norm mask
        easy_imageconcat = imageConcat()
        image_imageconcat = easy_imageconcat.concat(
            direction="right",
            match_image_size=match_image_size,
            image1=image1,
            image2=image2,
        )
        _, height1, width1, _ = image1.shape
        _, height2, width2, _ = image2.shape
        _, height, width, _ = image_imageconcat[0].shape
        # Create mask (0 for left image area, 1 for right image area)
        final_mask = torch.zeros((1, height, width))
        final_mask[:, :, width1:] = 1.0  # Set right half to 1
        
        # If mask is provided, subtract it from the right side
        if mask is not None:
            # Resize input mask to match height1
            pil_input_mask = tensor2pil(mask)
            pil_input_mask = pil_input_mask.resize((width-width1, height), Image.Resampling.LANCZOS)
            resized_input_mask = pil2tensor(pil_input_mask)
            # Subtract input mask from the right side
            if invert:
                final_mask[:, :, width1:] *= (1.0 - resized_input_mask)
            else:
                final_mask[:, :, width1:] *= resized_input_mask

        return (image_imageconcat[0], final_mask, )

class image_concat_mask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "mask": ("MASK",),
                "invert": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_concat_mask"
    CATEGORY = "cp🌻 Addoor/image"

    def image_concat_mask(self, image1, image2=None, mask=None, invert=True):
        processed_images = []
        masks = []
        
        for idx, img1 in enumerate(image1):
            # Convert tensor to PIL
            pil_image1 = tensor2pil(img1)
            
            # Get first image dimensions
            width1, height1 = pil_image1.size
            
            if image2 is not None and idx < len(image2):
                # Use provided second image
                pil_image2 = tensor2pil(image2[idx])
                width2, height2 = pil_image2.size
                
                # Resize image2 to match height of image1
                new_width2 = int(width2 * (height1 / height2))
                pil_image2 = pil_image2.resize((new_width2, height1), Image.Resampling.LANCZOS)
            else:
                # Create white image with same dimensions as image1
                pil_image2 = Image.new('RGB', (width1, height1), 'white')
                new_width2 = width1
            
            # Create new image to hold both images side by side
            combined_image = Image.new('RGB', (width1 + new_width2, height1))
            
            # Paste both images
            combined_image.paste(pil_image1, (0, 0))
            combined_image.paste(pil_image2, (width1, 0))
            
            # Convert combined image to tensor
            combined_tensor = pil2tensor(combined_image)
            processed_images.append(combined_tensor)
            
            # Create mask (0 for left image area, 1 for right image area)
            final_mask = torch.zeros((1, height1, width1 + new_width2))
            final_mask[:, :, width1:] = 1.0  # Set right half to 1
            
            # If mask is provided, subtract it from the right side
            if mask is not None and idx < len(mask):
                input_mask = mask[idx]
                # Resize input mask to match height1
                pil_input_mask = tensor2pil(input_mask)
                pil_input_mask = pil_input_mask.resize((new_width2, height1), Image.Resampling.LANCZOS)
                resized_input_mask = pil2tensor(pil_input_mask)
                
                # Subtract input mask from the right side
                if invert:
                    final_mask[:, :, width1:] *= (1.0 - resized_input_mask)
                else:
                    final_mask[:, :, width1:] *= resized_input_mask
            
            masks.append(final_mask)
            
        processed_images = torch.cat(processed_images, dim=0)
        masks = torch.cat(masks, dim=0)
        
        return (processed_images, masks)

