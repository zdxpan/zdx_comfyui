# layerstyle advance

import copy
import os.path
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image, ImageDraw
import cv2
from PIL import ImageFont
from .imagefunc import log

from folder_paths import models_dir

yolo_dir = os.path.join(models_dir, 'yolo')
yolo_dir2 = os.path.join(models_dir, 'ultralytics')

def pil2tensor(image):
    new_image = image.convert('RGB')
    img_array = np.array(new_image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)[None]
    return img_tensor

def tensor2pil(image):
    return Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8)) 

def np2pil(np_image:np.ndarray) -> Image:
    return Image.fromarray(np_image)

def image2mask(image:Image) -> torch.Tensor:
    image = image.convert('L')
    img_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(img_array)[None]

def add_mask(masks_a:torch.Tensor, masks_b:torch.Tensor) -> torch.Tensor:
    return torch.clamp(masks_a + masks_b, 0, 1)

class YoloV8Detect:

    def __init__(self):
        self.NODE_NAME = 'YoloV8Detect'


    @classmethod
    def INPUT_TYPES(self):
        yolo_models = os.listdir(yolo_dir)
        mask_merge = ["all", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        return {
            "required": {
                "image": ("IMAGE", ),
                "yolo_model": (yolo_models, {"default": "face_yolov8n.pt"}),
                "mask_merge": (mask_merge,),
            },
            "optional": {
                "conf": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "iou": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "classes": ("STRING", {"default": "", "multiline": False}),
                "device": ("STRING", {"default": "auto"}),
                "max_det": ("INT", {"default": 300, "min": 1, "max": 1000, "step": 1}),
                "retina_masks": ("BOOLEAN", {"default": True}),
                "agnostic_nms": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MASK" , "BOOLEAN")
    RETURN_NAMES = ("mask", "yolo_plot_image", "yolo_masks", "is_detect")
    FUNCTION = 'yolo_detect'
    CATEGORY = 'zdx/LayerUtility'

    def yolo_detect(self, image,
                          yolo_model, mask_merge,
                          conf=0.25, iou=0.45, classes="", device="auto", 
                          max_det=300, retina_masks=True, agnostic_nms=False
                      ):

        ret_masks = []
        ret_yolo_plot_images = []
        ret_yolo_masks = []

        from ultralytics import YOLO
        import ultralytics.nn.modules.head
        import ultralytics.nn.modules.block

        # Monkey patch for missing classes in custom/older models (e.g. yolo26)
        if not hasattr(ultralytics.nn.modules.head, 'Segment26'):
             setattr(ultralytics.nn.modules.head, 'Segment26', ultralytics.nn.modules.head.Segment)
        
        if not hasattr(ultralytics.nn.modules.head, 'Detect26'):
             setattr(ultralytics.nn.modules.head, 'Detect26', ultralytics.nn.modules.head.Detect)

        if not hasattr(ultralytics.nn.modules.block, 'Proto26'):
             setattr(ultralytics.nn.modules.block, 'Proto26', ultralytics.nn.modules.block.Proto)

        model_name = os.path.join(yolo_dir, yolo_model)
        extract_type = yolo_model.split('_')[0].split('-')[0]  #  'face_yolov8n.pt'-> face
        if not os.path.exists(model_name):
            model_name = os.path.join(yolo_dir2, yolo_model)
        if not os.path.exists(model_name):
            raise 'error, please put the yolo models in comfyui models/ultralytics'
        yolo_model = YOLO(model_name)
        
        # Parse classes parameter
        classes_list = None
        if classes and classes.strip():
            try:
                # Support comma-separated list of class IDs or ranges
                classes_list = []
                for item in classes.strip().split(','):
                    item = item.strip()
                    if '-' in item:  # Support range like "0-5"
                        start, end = map(int, item.split('-'))
                        classes_list.extend(range(start, end + 1))
                    else:
                        classes_list.append(int(item))
            except ValueError:
                log(f"{self.NODE_NAME} Invalid classes format: {classes}, ignoring.", message_type='warning')
                classes_list = None
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        log(f"{self.NODE_NAME} Using device: {device}, conf: {conf}, iou: {iou}, max_det: {max_det}")

        for i in image:
            i = torch.unsqueeze(i, 0)
            _image = tensor2pil(i)
            results = yolo_model(_image, 
                               conf=conf, 
                               iou=iou, 
                               classes=classes_list,
                               device=device,
                               max_det=max_det,
                               retina_masks=retina_masks,
                               agnostic_nms=agnostic_nms)
            
            # Reset masks for each image in batch
            current_image_masks = []
            
            for result in results:
                yolo_plot_image = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                ret_yolo_plot_images.append(pil2tensor(Image.fromarray(yolo_plot_image)))
                # have mask
                if result.masks is not None and len(result.masks) > 0:
                    masks_data = result.masks.data
                    for index, mask in enumerate(masks_data):
                        _mask = mask.cpu().numpy() * 255
                        _mask = np2pil(_mask).convert("L")
                        current_image_masks.append(image2mask(_mask))
                # no mask, if have box, draw box
                elif result.boxes is not None and len(result.boxes.xyxy) > 0:
                    white_image = Image.new('L', _image.size, "white")
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        _mask = Image.new('L', _image.size, "black")
                        _mask.paste(white_image.crop((x1, y1, x2, y2)), (x1, y1))
                        current_image_masks.append(image2mask(_mask))
                # no mask and box, add a black mask
                else:
                    current_image_masks.append(torch.zeros((1, _image.size[1], _image.size[0]), dtype=torch.float32))
                    # ret_yolo_masks.append(image2mask(Image.new('L', _image.size, "black")))
                    log(f"{self.NODE_NAME} mask or box not detected.")

                # merge mask for current image
                if not current_image_masks:
                     # Fallback empty mask if list is somehow empty
                     _mask = torch.zeros((1, _image.size[1], _image.size[0]), dtype=torch.float32)
                else:
                    _mask = current_image_masks[0]
                    if mask_merge == "all":
                        for i in range(len(current_image_masks) - 1):
                            _mask = add_mask(_mask, current_image_masks[i + 1])
                    else:
                        for i in range(min(len(current_image_masks), int(mask_merge)) - 1):
                            _mask = add_mask(_mask, current_image_masks[i + 1])
                
                ret_masks.append(_mask)
                ret_yolo_masks.extend(current_image_masks)

        log(f"{self.NODE_NAME} Processed {len(ret_masks)} image(s).", message_type='finish')
        final_mask = torch.cat(ret_masks, dim=0)
        is_empty = torch.all(final_mask == 0)
        return (final_mask,
                torch.cat(ret_yolo_plot_images, dim=0),
                torch.cat(ret_yolo_masks, dim=0), 
                not is_empty)

# NODE_CLASS_MAPPINGS = {
#     "LayerMask: YoloV8Detect": YoloV8Detect
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "LayerMask: YoloV8Detect": "LayerMask: YoloV8 Detect(Advance)"
# }
