import os
import torch
import numpy as np
from ultralytics import YOLO, SAM
import requests
import json
import comfy
from torchvision import transforms
import torch.nn.functional as F
from nodes import MAX_RESOLUTION
import torchvision
from PIL import Image, ImageDraw
import cv2
from PIL import ImageFont


from nodes import NODE_CLASS_MAPPINGS

from folder_paths import models_dir

from .layerstyle.imagefunc import chop_image

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
    if image.mode == 'L':
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    else:
        image = image.convert('RGB').split()[0]
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
def add_mask(masks_a:torch.Tensor, masks_b:torch.Tensor) -> torch.Tensor:
    mask = chop_image(tensor2pil(masks_a), tensor2pil(masks_b), blend_mode='add', opacity=100)
    return image2mask(mask)
yolo_dir = os.path.join(models_dir, 'yolo')
yolo_dir2 = os.path.join(models_dir, 'ultralytics')

coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
_COLORS = (
    np.array(
        [
            0.000,0.447,0.741,0.850,0.325,0.098,0.929,0.694,0.125,0.494,0.184,0.556,0.466,0.674,0.188,0.301,0.745,0.933,0.635,0.078,0.184,
            0.300,0.300,0.300,0.600,0.600,0.600,1.000,0.000,0.000,1.000,0.500,0.000,0.749,0.749,0.000,0.000,1.000,0.000,0.000,0.000,1.000,
            0.667,0.000,1.000,0.333,0.333,0.000,0.333,0.667,0.000,0.333,1.000,0.000,0.667,0.333,0.000,0.667,0.667,0.000,0.667,1.000,0.000,
            1.000,0.333,0.000,1.000,0.667,0.000,1.000,1.000,0.000,0.000,0.333,0.500,0.000,0.667,0.500,0.000,1.000,0.500,0.333,0.000,0.500,
            0.333,0.333,0.500,0.333,0.667,0.500,0.333,1.000,0.500,0.667,0.000,0.500,0.667,0.333,0.500,0.667,0.667,0.500,0.667,1.000,0.500,
            1.000,0.000,0.500,1.000,0.333,0.500,1.000,0.667,0.500,1.000,1.000,0.500,0.000,0.333,1.000,0.000,0.667,1.000,0.000,1.000,1.000,
            0.333,0.000,1.000,0.333,0.333,1.000,0.333,0.667,1.000,0.333,1.000,1.000,0.667,0.000,1.000,0.667,0.333,1.000,0.667,0.667,1.000,
            0.667,1.000,1.000,1.000,0.000,1.000,1.000,0.333,1.000,1.000,0.667,1.000,0.333,0.000,0.000,0.500,0.000,0.000,0.667,0.000,0.000,
            0.833,0.000,0.000,1.000,0.000,0.000,0.000,0.167,0.000,0.000,0.333,0.000,0.000,0.500,0.000,0.000,0.667,0.000,0.000,0.833,0.000,
            0.000,1.000,0.000,0.000,0.000,0.167,0.000,0.000,0.333,0.000,0.000,0.500,0.000,0.000,0.667,0.000,0.000,0.833,0.000,0.000,1.000,
            0.000,0.000,0.000,0.143,0.143,0.143,0.286,0.286,0.286,0.429,0.429,0.429,0.571,0.571,0.571,0.714,0.714,0.714,0.857,0.857,0.857,
            0.000,0.447,0.741,0.314,0.717,0.741,0.50,0.5,0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)

def get_yolo_result(results):
    res = {}
    if len(results) < 1:
        return None
    it = results[0]
    img = np.copy(it.orig_img)
    height, width = img.shape[:2]
    for ci, c in enumerate(it):
        # c.names: {0: 'person'}
        #  Get detection class name
        label = c.names[c.boxes.cls.tolist().pop()]  # person
        bbox = c.boxes.xywh.cpu().tolist().pop()
        bbox_n = c.boxes.xyxyn.cpu().tolist().pop()
        bbox_xy = c.boxes.xyxy.cpu().tolist().pop()
        # Create binary mask
        b_mask = np.zeros(img.shape[:2], np.uint8)
        #  Extract contour result
        b_mask_pil = None
        if c.masks is not None:
            contour = c.masks.xy.pop()
            #  Changing the type
            contour = contour.astype(np.int32)
            #  Reshaping
            contour = contour.reshape(-1, 1, 2)
            # Draw contour onto mask
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            b_mask_pil = Image.fromarray(b_mask)
        # b_mask_pil.save('/home/dell/study/test_comfy/img/yolo_person_mask.png')
        item = {'mask': b_mask_pil, 'bbox': bbox, 'bbox_n': bbox_n, 'bbox_xy': bbox_xy}
        if label not in res:
            res[label] =  [item]
        else:
            res[label].append(item)
    return res


class BBoxVisNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BOXES",),
                "category_ids": ("LABELS", {"default": "None"}),
                "rect_size": ("INT", {"default": 3, "min": 0, "step": 1}),
                "text_size": ("INT", {"default": 2, "min": 0, "step": 1}),
                "font_scale": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.1}),
                "show_label": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_bbox"

    CATEGORY = "Ultralytics/Utils"


    def draw_bbox(self, image, bboxes, category_ids, font_scale, rect_size=None, text_size=None, show_label=True):
        if image.dim() == 4 and image.size(0) == 1:
            image = image.squeeze(0)
        
        image = image.cpu().numpy()
        
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        for index in range(len(bboxes)):
            category_name = coco_classes
            category_id = int(category_ids[index])

            rect_size = rect_size or max(round(sum(image.shape) / 2 * 0.001), 1)
            text_size = text_size or max(rect_size - 1, 1)

            color = (_COLORS[category_id] * 255).astype(np.uint8).tolist()
            text = f"{category_name[category_id]}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, font_scale, text_size)[0]
            txt_color = (0, 0, 0) if np.mean(_COLORS[category_id]) > 0.5 else (255, 255, 255)

            x, y, w, h = bboxes[index]
            x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, rect_size)
            txt_bk_color = (_COLORS[category_id] * 255 * 0.7).astype(np.uint8).tolist()

            if show_label:
                cv2.rectangle(
                    image,
                    (x1, y1 + 1),
                    (x1 + txt_size[0] + 1, y1 + int(1.5 * txt_size[1])),
                    txt_bk_color,
                    -1,
                )
                cv2.putText(
                    image,
                    text,
                    (x1, y1 + txt_size[1]),
                    font,
                    font_scale,
                    txt_color,
                    thickness=text_size,
                )
        tensor_image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        
        
        return (tensor_image,)

class GetImageSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("Height", "Width")
    FUNCTION = "get_image_size"

    CATEGORY = "Ultralytics/Utils"

    def get_image_size(self, image):
        return (image.shape[1], image.shape[2],)

class ImageResizeAdvanced:
    # https://github.com/cubiq/ComfyUI_essentials/blob/main/image.py
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 32, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 32, }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "method": (["stretch", "keep proportion", "fill / crop", "pad"],),
                "condition": (["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"],),
                "multiple_of": ("INT", { "default": 0, "min": 0, "max": 512, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "Ultralytics/Utils"

    def execute(self, image, width, height, method="stretch", interpolation="nearest", condition="always", multiple_of=0, keep_proportion=False):
        _, oh, ow, _ = image.shape
        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion:
            method = "keep proportion"

        if multiple_of > 1:
            width = width - (width % multiple_of)
            height = height - (height % multiple_of)

        if method == 'keep proportion' or method == 'pad':
            if width == 0 and oh < height:
                width = MAX_RESOLUTION
            elif width == 0 and oh >= height:
                width = ow

            if height == 0 and ow < width:
                height = MAX_RESOLUTION
            elif height == 0 and ow >= width:
                height = ow

            ratio = min(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)

            if method == 'pad':
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        elif method.startswith('fill'):
            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = max(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)
            x = (new_width - width) // 2
            y = (new_height - height) // 2
            x2 = x + width
            y2 = y + height
            if x2 > new_width:
                x -= (x2 - new_width)
            if x < 0:
                x = 0
            if y2 > new_height:
                y -= (y2 - new_height)
            if y < 0:
                y = 0
            width = new_width
            height = new_height
        else:
            width = width if width > 0 else ow
            height = height if height > 0 else oh

        if "always" in condition \
            or ("downscale if bigger" == condition and (oh > height or ow > width)) or ("upscale if smaller" == condition and (oh < height or ow < width)) \
            or ("bigger area" in condition and (oh * ow > height * width)) or ("smaller area" in condition and (oh * ow < height * width)):

            outputs = image.permute(0,3,1,2)

            if interpolation == "lanczos":
                outputs = comfy.utils.lanczos(outputs, width, height)
            else:
                outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

            if method == 'pad':
                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

            outputs = outputs.permute(0,2,3,1)

            if method.startswith('fill'):
                if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                    outputs = outputs[:, y:y2, x:x2, :]
        else:
            outputs = image

        if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
            width = outputs.shape[2]
            height = outputs.shape[1]
            x = (width % multiple_of) // 2
            y = (height % multiple_of) // 2
            x2 = width - ((width % multiple_of) - x)
            y2 = height - ((height % multiple_of) - y)
            outputs = outputs[:, y:y2, x:x2, :]

        return(outputs, outputs.shape[2], outputs.shape[1],)

class CocoToNumber:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coco_label": (
                    coco_classes,
                    {"default": "person"},
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "map_class"
    CATEGORY = "zdx/Utils"

    def map_class(self, coco_label):
        class_num = str(coco_classes.index(coco_label))
        return (class_num,)

class UltralyticsModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model_name": (
                    [
                        "yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt", "yolov5lu.pt", "yolov5xu.pt",
                        "yolov5n6u.pt", "yolov5s6u.pt", "yolov5m6u.pt", "yolov5l6u.pt", "yolov5x6u.pt",
                        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
                        "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt",
                        "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt",
                        
                    ],
                ),
            },
        }

    RETURN_TYPES = ("ULTRALYTICS_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "zdx/Model"

    def coco_to_labels(self, coco):
        labels = []
        for category in coco["categories"]:
            labels.append(category["name"])
        return labels

class CustomUltralyticsModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        models_dir = "comfyui/models/ultralytics"  # Update with the appropriate directory
        files = []
        for root, dirs, filenames in os.walk(models_dir):
            for filename in filenames:
                if filename.endswith(".pt"):
                    relative_path = os.path.relpath(os.path.join(root, filename), models_dir)
                    files.append(relative_path)
        return {
            "required": {
                "model_path": (sorted(files), {"model_upload": True})
            }
        }

    CATEGORY = "Ultralytics/Model"
    RETURN_TYPES = ("ULTRALYTICS_MODEL",)
    FUNCTION = "load_model"

    def load_model(self, model_path):
        model_full_path = os.path.join("ComfyUI/models/ultralytics", model_path)  # Update with the appropriate directory
        model = YOLO(model_full_path)
        return (model,)

class UltralyticsModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model_name": (
                    [
                        "yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt", "yolov5lu.pt", "yolov5xu.pt",
                        "yolov5n6u.pt", "yolov5s6u.pt", "yolov5m6u.pt", "yolov5l6u.pt", "yolov5x6u.pt",
                        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
                        "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt",
                        "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt",
                        "mobile_sam.pt"
                    ],
                ),
            },
        }

    RETURN_TYPES = ("ULTRALYTICS_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "Ultralytics/Model"

    def __init__(self):
        self.loaded_models = {}

    def load_model(self, model_name=None):
        if model_name is None:
            model_name = "yolov8s.pt"  # Default model name if not provided

        if model_name in self.loaded_models:
            print(f"Model {model_name} already loaded. Returning cached model.")
            return (self.loaded_models[model_name],)

        model_url = f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{model_name}"

        # Create a "models/ultralytics" directory if it doesn't exist
        os.makedirs(os.path.join("ComfyUI", "models", "ultralytics"), exist_ok=True)

        model_path = os.path.join("ComfyUI", "models", "ultralytics", model_name)

        # Check if the model file already exists
        if os.path.exists(model_path):
            print(f"Model {model_name} already downloaded. Loading model.")
        else:
            print(f"Downloading model {model_name}...")
            response = requests.get(model_url)
            response.raise_for_status()  # Raise an exception if the download fails

            with open(model_path, "wb") as file:
                file.write(response.content)

            print(f"Model {model_name} downloaded successfully.")

        model = YOLO(model_path)
        self.loaded_models[model_name] = model
        return (model,)

class BBoxToCoco:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "results": ("ULTRALYTICS_RESULTS",),
                "bbox": ("BOXES",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("coco_json",)
    FUNCTION = "bbox_to_xywh"
    OUTPUT_NODE = True

    CATEGORY = "Ultralytics/Utils"

    def bbox_to_xywh(self, results, bbox):
        coco_data = {
            "categories": [],
            "images": [],
            "annotations": [],
        }

        annotation_id = 1
        category_names = results[0].names

        if isinstance(bbox, list):
            for frame_idx, bbox_frame in enumerate(bbox):
                image_id = frame_idx + 1
                image_width, image_height = results[frame_idx].boxes.orig_shape[1], results[frame_idx].boxes.orig_shape[0]
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": f"{image_id:04d}.jpg",
                    "height": image_height,
                    "width": image_width,
                })

                for bbox_single, cls_single in zip(bbox_frame, results[frame_idx].boxes.cls):
                    x = float(bbox_single[0])
                    y = float(bbox_single[1])
                    w = float(bbox_single[2])
                    h = float(bbox_single[3])
                    category_id = int(cls_single.item()) + 1

                    if category_id not in [cat["id"] for cat in coco_data["categories"]]:
                        coco_data["categories"].append({
                            "id": category_id,
                            "name": category_names[category_id - 1],
                            "supercategory": "none"
                        })

                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "segmentation": [],
                        "iscrowd": 0
                    })
                    annotation_id += 1
        else:
            image_id = 1
            image_width, image_height = results[0].boxes.orig_shape[1], results[0].boxes.orig_shape[0]
            coco_data["images"].append({
                "id": image_id,
                "file_name": f"{image_id:04d}.jpg",
                "height": image_height,
                "width": image_width,
            })

            for bbox_single, cls_single in zip(bbox, results[0].boxes.cls):
                if bbox_single.dim() == 0:
                    x = float(bbox_single.item())
                    y = float(bbox_single.item())
                    w = float(bbox_single.item())
                    h = float(bbox_single.item())
                else:
                    x = float(bbox_single[0])
                    y = float(bbox_single[1])
                    w = float(bbox_single[2])
                    h = float(bbox_single[3])

                category_id = int(cls_single.item()) + 1

                if category_id not in [cat["id"] for cat in coco_data["categories"]]:
                    coco_data["categories"].append({
                        "id": category_id,
                        "name": category_names[category_id - 1],
                        "supercategory": "none"
                    })

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "segmentation": [],
                    "iscrowd": 0
                })
                annotation_id += 1

        coco_json = json.dumps(coco_data, indent=2)
        return (coco_json,)

class BBoxToXYWH:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ("INT", {"default": 0}),
                "bbox": ("BOXES", {"default": None}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("StrBox", "BOXES","X_coord", "Y_coord", "Width", "Height",)
    FUNCTION = "bbox_to_xywh"
    OUTPUT_NODE = True

    CATEGORY = "Ultralytics/Utils"

    def bbox_to_xywh(self, index, bbox):
        bbox = bbox[index]

        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])

        fullstr = f"x: {x}, y: {y}, w: {w}, h: {h}"

        return (fullstr,bbox, x,y,w,h,)

class ConvertToDict:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "bbox": ("BOXES", {"default": None}),
                "mask": ("MASKS", {"default": None}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert_to_dict"
    OUTPUT_NODE = True

    CATEGORY = "Ultralytics/Utils"

    def convert_to_dict(self, bbox=None, mask=None):
        output = {"objects": []}

        if bbox is not None:
            for obj_bbox in bbox:
                bbox_dict = {
                    "x": obj_bbox[0].item(),
                    "y": obj_bbox[1].item(),
                    "width": obj_bbox[2].item(),
                    "height": obj_bbox[3].item()
                }
                output["objects"].append({"bbox": bbox_dict})

        if mask is not None:
            for obj_mask in mask:
                mask_dict = {
                    "shape": obj_mask.shape,
                    "data": obj_mask.tolist()
                }
                if len(output["objects"]) > len(mask):
                    output["objects"].append({"mask": mask_dict})
                else:
                    output["objects"][-1]["mask"] = mask_dict

        if not output["objects"]:
            output = {"message": "No input provided"}

        import json
        output_str = json.dumps(output, indent=2)

        return {"ui": {"text": output_str}, "result": (output_str,)}

class UltralyticsInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("ULTRALYTICS_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "conf": ("FLOAT", {"default": 0.25, "min": 0, "max": 1, "step": 0.01}),
                "iou": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.01}),
                "height": ("INT", {"default": 640, "min": 64, "max": 1280, "step": 32}),
                "width": ("INT", {"default": 640, "min": 64, "max": 1280, "step": 32}),
                "device":(["cuda:0", "cpu"], ),
                "half": ("BOOLEAN", {"default": False}),
                "augment": ("BOOLEAN", {"default": False}),
                "agnostic_nms": ("BOOLEAN", {"default": False}),
                "classes": ("STRING", {"default": "None"}),

            },
        }
    RETURN_TYPES = ("ULTRALYTICS_RESULTS","IMAGE", "BOXES", "MASKS", "PROBABILITIES", "KEYPOINTS", "OBB", "LABELS",)
    FUNCTION = "inference"
    CATEGORY = "Ultralytics/Inference"

    def inference(self, model, image, conf=0.25, iou=0.7, height=640, width=640, device="cuda:0", half=False, augment=False, agnostic_nms=False, classes=None):
        if classes == "None":
            class_list = None
        else:
            class_list = [int(cls.strip()) for cls in classes.split(',')]

        if image.shape[0] > 1:
            batch_size = image.shape[0]
            results = []
            for i in range(batch_size):
                yolo_image = image[i].unsqueeze(0).permute(0, 3, 1, 2)
                result = model.predict(yolo_image, conf=conf, iou=iou, imgsz=(height, width), device=device, half=half, augment=augment, agnostic_nms=agnostic_nms, classes=class_list)
                results.append(result)

            boxes = [result[0].boxes.xywh for result in results]
            masks = [result[0].masks for result in results]
            probs = [result[0].probs for result in results]
            keypoints = [result[0].keypoints for result in results]
            obb = [result[0].obb for result in results]
            labels = [result[0].boxes.cls.cpu().tolist() for result in results]

        else:
            yolo_image = image.permute(0, 3, 1, 2)
            results = model.predict(yolo_image, conf=conf, iou=iou, imgsz=(height,width), device=device, half=half, augment=augment, agnostic_nms=agnostic_nms, classes=class_list)

            boxes = results[0].boxes.xywh
            masks = results[0].masks
            probs = results[0].probs
            keypoints = results[0].keypoints
            obb = results[0].obb     
            labels = results[0].boxes.cls.cpu().tolist()     

        return (results, image, boxes, masks, probs, keypoints, obb, labels,)


class UltralyticsVisualization:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "results": ("ULTRALYTICS_RESULTS",),
                "image": ("IMAGE",),
                "line_width": ("INT", {"default": 3}),
                "font_size": ("INT", {"default": 1}),
                "sam": ("BOOLEAN", {"default": True}),
                "kpt_line": ("BOOLEAN", {"default": True}),
                "labels": ("BOOLEAN", {"default": True}),
                "boxes": ("BOOLEAN", {"default": True}),
                "masks": ("BOOLEAN", {"default": True}),
                "probs": ("BOOLEAN", {"default": True}),
                "color_mode": (["class", "instance"], {"default": "class"}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize"
    CATEGORY = "Ultralytics/Vis"

    # ref: https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results.plot
    def visualize(self, image, results, line_width=3, font_size=1, sam=True, kpt_line=True, labels=True, boxes=True, masks=True, probs=True, color_mode="class"):
        if image.shape[0] > 1:
            batch_size = image.shape[0]
            annotated_frames = []
            for result in results:
                for r in result:
                    im_bgr = r.plot(im_gpu=True, line_width=line_width, font_size=font_size, kpt_line=kpt_line, labels=labels, boxes=boxes, masks=masks, probs=probs, color_mode=color_mode) 
                    annotated_frames.append(im_bgr)

            tensor_image = torch.stack([torch.from_numpy(np.array(frame).astype(np.float32) / 255.0) for frame in annotated_frames])

        else:
            annotated_frames = []
            for r in results:
                if sam == True:
                    im_bgr = r.plot(line_width=line_width, font_size=font_size, kpt_line=kpt_line, labels=labels, boxes=boxes, masks=masks, probs=probs, color_mode=color_mode)  # BGR-order numpy array

                else:
                    im_bgr = r.plot(im_gpu=True, line_width=line_width, font_size=font_size, kpt_line=kpt_line, labels=labels, boxes=boxes, masks=masks, probs=probs, color_mode=color_mode)  # BGR-order numpy array
                annotated_frames.append(im_bgr)

            tensor_image = torch.stack([torch.from_numpy(np.array(frame).astype(np.float32) / 255.0) for frame in annotated_frames])

        return (tensor_image,)

class ViewText:
    # https://github.com/gokayfem/ComfyUI_VLM_nodes/blob/main/nodes/simpletext.py
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "view_text"
    OUTPUT_NODE = True

    CATEGORY = "Ultralytics/Utils"

    def view_text(self, text):
        # Parse the combined JSON string
        return {"ui": {"text": text}, "result": (text,)}

def sort_res_by_bbox(res):
    for k in res:
        res[k] = sorted(res[k], key=lambda x: (x['bbox'][2] * x['bbox'][3]), reverse=True)
    return res


def yolo_detect(image, detec_type = 'face', confidence=0.5, debug=False):
    yolo_infer = UltralyticsInference()
    yolo_viser = UltralyticsVisualization()
    model_config = {'face': 'face_yolov8n.pt', 'body': 'person_yolov8n-seg.pt', 'person': 'person_yolov8n-seg.pt', 
                    'hand': 'hand_yolov8n.pt', 'foot': 'foot-yolov8l.pt'}
    if detec_type not in model_config:
        raise f'detect_type is not surppott, current only surpport:{model_config}'
    model_path = model_config[detec_type]
    model_name = os.path.join(yolo_dir, model_path)
    if not os.path.exists(model_name):
        model_name = os.path.join(yolo_dir2, model_path)
    if not os.path.exists(model_name):
        raise 'error, please put the yolo models in comfyui models/ultralytics'
    yolo_model = YOLO(model_name)
    yres = yolo_infer.inference(model=yolo_model,conf = confidence, iou=0.7, image=image, classes="None")
    res = get_yolo_result(yres[0])   # get label -> [mask, bbox]
    res = sort_res_by_bbox(res)   # sort by bbox area, from big to small
    if debug:
        yolo_vis1 = tensor2pil(
            yolo_viser.visualize(image, yres[0])[0]
        )
        res['debug_image'] = yolo_vis1
    return res


class MainObjDetect:
    '''
    focus process the object, even can work in tile mode~ 
    return: bbox_normalized, can apply in any aespecratio relatively, return image as pil
    use_example~  maybe rembg will enhance ~
    '''
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "extract_type": (['person', 'body', 'face', 'hands', 'shoe'], {"default": "person"}),
            },
        }
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "detect"
    CATEGORY = "zdx/Logic"
    def __init__(self):
        self.extract_types = ['person', 'body', 'face', 'hands', 'shoe']
        # self.imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()

    def detect(self, input_image, extract_type = 'person'):
        # yolo small model work better in small area~
        input_image_tensor = input_image
        if not isinstance(input_image, torch.Tensor):
            input_image_tensor = pil2tensor(input_image)
        
        image_pil = tensor2pil(input_image_tensor) if isinstance(input_image_tensor, torch.Tensor) else input_image_tensor
        w, h = image_pil.size
        image_pil = image_pil.resize((w//32*32, h//32*32))
        input_image_tensor = pil2tensor(image_pil)

        detect_res = yolo_detect(input_image_tensor, detec_type = extract_type, debug=False)

        return (detect_res is not None and extract_type in detect_res, )


class YoloDetect:
    '''
    focus process the object, even can work in tile mode~ 
    return: bbox_normalized, can apply in any aespecratio relatively, return image as pil
    use_example~  maybe rembg will enhance ~
    '''
    @classmethod
    def INPUT_TYPES(s):
        yolo_models = os.listdir(yolo_dir)
        return {
            "required": {
                "image": ("IMAGE",),
                "confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "yolo_model": (yolo_models, {"default": "face_yolov8n.pt"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("BOOLEAN", "IMAGE", "MASK")
    RETURN_NAMES = ("boolean", "visual_image", "masks")
    FUNCTION = "forward"
    CATEGORY = "zdx/logic"
    def __init__(self):
        self.yolo_infer = UltralyticsInference()
        self.yolo_viser = UltralyticsVisualization()

    def forward(self, image, yolo_model, confidence=0.5, debug=False):
        model_config = {'face': 'face_yolov8n.pt', 'body': 'person_yolov8n-seg.pt', 'person': 'person_yolov8n-seg.pt', 
                        'hand': 'hand_yolov8n.pt', 'foot': 'foot-yolov8l.pt'}
        model_name = os.path.join(yolo_dir, yolo_model)
        extract_type = yolo_model.split('_')[0].split('-')[0]
        if extract_type not in model_config:
            raise f'error, yolo model {yolo_model} not surpport, current only surpport:{model_config}'
        if not os.path.exists(model_name):
            model_name = os.path.join(yolo_dir2, yolo_model)
        if not os.path.exists(model_name):
            raise 'error, please put the yolo models in comfyui models/ultralytics'
        yolo_model = YOLO(model_name)
        
        image_pil = tensor2pil(image) if isinstance(image, torch.Tensor) else image
        w,h = image_pil.size
        ret_yolo_masks = []
        if max(w,h) > 2048:
            image_pil = image_pil.resize((w*2048//max(w,h), h*2048//max(w,h)))
        # resize到整除32
        image_pil = image_pil.resize((w//32*32, h//32*32))
        input_image_tensor = pil2tensor(image_pil)
        # device 如果是mps 没有cuda则使用cpu
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        yres = self.yolo_infer.inference(model=yolo_model, conf = confidence, iou=0.7, image=input_image_tensor, classes="None", device=device)
        res = get_yolo_result(yres[0])   # get label -> [mask, bbox]
        res = sort_res_by_bbox(res)   # sort by bbox area, from big to small
        # mask visual
        result = yres[0]
        # if len(result) > 0:
        for item in result:        
            if item.masks is not None and len(item.masks) > 0:
                masks = []
                masks_data = item.masks.data
                for index, mask in enumerate(masks_data):
                    _mask = mask.cpu().numpy() * 255
                    _mask = np2pil(_mask).convert("L")
                    ret_yolo_masks.append(image2mask(_mask))
        # merge mask
        yolo_masks = []
        if len(ret_yolo_masks) > 0:
            yolo_masks.append(ret_yolo_masks[0])
            for i in range(1, len(ret_yolo_masks)):
                yolo_masks.append(add_mask(yolo_masks[-1], ret_yolo_masks[i]))
        else:
            yolo_masks.append(torch.zeros((1, h, w), dtype=torch.float32))
            
        yolo_masks = torch.cat(yolo_masks, dim=0)
        yolo_vis1 = image
        if debug:
            yolo_vis1 = tensor2pil(
                self.yolo_viser.visualize(image, yres[0])[0]
            )
            yolo_vis1 = pil2tensor(yolo_vis1)
            res['debug_image'] = yolo_vis1
        # return res
        is_detected = res is not None and extract_type in res
        return (is_detected, yolo_vis1, yolo_masks)


def expand_bbox(bbox, image_width, image_height, expand_ratio=0.1, width_more = False):
    """
    扩展bbox的大小，同时确保不超出图像边界。
    参数:
    bbox: 列表或元组，格式为 [x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio]，为归一化的比率形式。
    image_width: 图像宽度（像素）
    image_height: 图像高度（像素）
    expand_ratio: 扩展比例，默认为0.1（10%）
    返回:
    expanded_bbox: 扩展后的bbox，格式与输入相同（归一化的比率形式）
    """
    assert bbox[0] <= 1.0 or bbox[1] <= 1.0 or bbox[2] <= 1.0 or bbox[3] <= 1.0
    x_min = bbox[0] * image_width
    y_min = bbox[1] * image_height
    x_max = bbox[2] * image_width
    y_max = bbox[3] * image_height
    
    width = x_max - x_min
    height = y_max - y_min

    width_expand = width * expand_ratio
    height_expand = height * expand_ratio

    if width_more and (height / width > 1.4):
        width_expand = width * (expand_ratio  + 0.4)
        if height / width > 1.6:
            width_expand = width * (expand_ratio  + 0.7)

    new_x_min = max(0, x_min - width_expand / 2)  # 确保不小于0
    new_y_min = max(0, y_min - height_expand / 2)  # 确保不小于0
    new_x_max = min(image_width, x_max + width_expand / 2)  # 确保不超过图像宽度
    new_y_max = min(image_height, y_max + height_expand / 2)  # 确保不超过图像高度
    
    expanded_bbox = [
        new_x_min / image_width,
        new_y_min / image_height,
        new_x_max / image_width,
        new_y_max / image_height
    ]
    expanded_bbox_mx = [
        new_x_min, new_y_min,
        new_x_max, new_y_max
    ]
    
    return expanded_bbox, expanded_bbox_mx

class MainObjExtract:
    '''
    focus process the object, even can work in tile mode~ 
    return: bbox_normalized, can apply in any aespecratio relatively, return image as pil
    use_example~  maybe rembg will enhance ~
    '''
    @classmethod
    def INPUT_TYPES(s):
        yolo_models = os.listdir(yolo_dir)
        return {
            "required": {
                "image": ("IMAGE",),
                "confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "yolo_model": (yolo_models, {"default": "face_yolov8n.pt"}),
                "debug": ("BOOLEAN", {"default": False}),
                "expand_rate": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0}),
            },
        }
    RETURN_TYPES = ( "IMAGE", "BBOX", "BBOX", "IMAGE")
    RETURN_NAMES = ("image", "bbox_normalized", "bbox_absolute", "visual_image",)
    FUNCTION = "forward"
    CATEGORY = "zdx/logic"
    def __init__(self):
        self.yolo_infer = UltralyticsInference()
        self.yolo_viser = UltralyticsVisualization()

    def forward(self, image, yolo_model, confidence=0.5, debug=False, expand_rate=0.5):
        model_config = {'face': 'face_yolov8n.pt', 'body': 'person_yolov8n-seg.pt', 'person': 'person_yolov8n-seg.pt', 
                        'hand': 'hand_yolov8n.pt', 'foot': 'foot-yolov8l.pt'}
        model_name = os.path.join(yolo_dir, yolo_model)
        extract_type = yolo_model.split('_')[0].split('-')[0]
        if extract_type not in model_config:
            raise f'error, yolo model {yolo_model} not surpport, current only surpport:{model_config}'
        if not os.path.exists(model_name):
            model_name = os.path.join(yolo_dir2, yolo_model)
        if not os.path.exists(model_name):
            raise 'error, please put the yolo models in comfyui models/ultralytics'
        yolo_model = YOLO(model_name)
        
        image_pil = tensor2pil(image) if isinstance(image, torch.Tensor) else image
        human_img_crop_enhanced = image_pil
        w,h = image_pil.size
        # if max(w,h) > 2048:
        #     image_pil = image_pil.resize((w*2048//max(w,h), h*2048//max(w,h)))
        # resize到整除32
        image_pil = image_pil.resize((w//32*32, h//32*32))
        input_image_tensor = pil2tensor(image_pil)
        w,h = image_pil.size
        
        yres = self.yolo_infer.inference(model=yolo_model, conf = confidence, iou=0.7, image=input_image_tensor, classes="None")
        res = get_yolo_result(yres[0])   # get label -> [mask, bbox]
        res = sort_res_by_bbox(res)   # sort by bbox area, from big to small

        # yres = yolo_detect(input_image_tensor, detec_type = 'body', debug=True)
        if res is not None and extract_type in res:
            box_mask = res[extract_type][0]   #  choose the max area size 
            box_mask_mask = box_mask['mask']
            bbox_normal = box_mask['bbox_n']
            # bx = box_mask['bbox_xy']
            # Human Bbox expand. Must half of orig h w 
            bbox_normal_expand, bbox_expand  = expand_bbox(bbox=bbox_normal, image_width=w, image_height=h, expand_ratio=expand_rate, width_more=True)
            # width, height = bbox_expand[2] - bbox_expand[0], bbox_expand[3] - bbox_expand[1]
            # if width / W < 0.6  or  height / H < 0.6:
            ORIG_BBOX_NORMAL = bbox_normal_expand
            ORIG_BBOX = [int(x_) for x_ in bbox_expand]
            human_img_crop_enhanced = image_pil.crop(ORIG_BBOX)
        yolo_vis1 = image
        if debug:
            yolo_vis1 = tensor2pil(
                self.yolo_viser.visualize(image, yres[0])[0]
            )
            yolo_vis1 = pil2tensor(yolo_vis1)
        
        return pil2tensor(human_img_crop_enhanced), ORIG_BBOX_NORMAL, ORIG_BBOX, yolo_vis1

class YoloBodyPoseCheck:
    def __init__(self, model_path="/data/models/yolo/yolov8n-pose.pt"):
        self.model = YOLO(model_path)
        self.KEYPOINTS = {
            "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
            "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
            "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
            "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
        }
        
        # 定义关键点之间的连接关系（用于绘制骨架）
        self.SKELETON = [
            # 头部连接
            [0, 1], [0, 2], [1, 3], [2, 4],  # nose->eyes->ears
            # 躯干连接
            [5, 6], [5, 11], [6, 12], [11, 12],  # shoulders, hips
            # 左臂
            [5, 7], [7, 9],  # left_shoulder->left_elbow->left_wrist
            # 右臂
            [6, 8], [8, 10],  # right_shoulder->right_elbow->right_wrist
            # 左腿
            [11, 13], [13, 15],  # left_hip->left_knee->left_ankle
            # 右腿
            [12, 14], [14, 16],  # right_hip->right_knee->right_ankle
        ]
        
        # 关键点颜色（BGR格式）
        self.KP_COLORS = [
            (255, 0, 0),    # nose - 红色
            (255, 85, 0),   # left_eye
            (255, 170, 0),  # right_eye
            (255, 255, 0), # left_ear
            (170, 255, 0), # right_ear
            (85, 255, 0),  # left_shoulder
            (0, 255, 0),   # right_shoulder - 绿色
            (0, 255, 85), # left_elbow
            (0, 255, 170),# right_elbow
            (0, 255, 255),# left_wrist
            (0, 170, 255),# right_wrist
            (0, 85, 255), # left_hip
            (0, 0, 255),  # right_hip - 蓝色
            (85, 0, 255), # left_knee
            (170, 0, 255),# right_knee
            (255, 0, 255),# left_ankle
            (255, 0, 170),# right_ankle
        ]
        
        # 连接线颜色（BGR格式）
        self.LINE_COLOR = (0, 255, 255)  # 黄色

    def check_half_person_from_array(self, image_bgr, knee_conf=0.5, ankle_conf=0.8):
        """
        从BGR格式的numpy数组检测姿态
        """
        # conf 参数：检测置信度阈值（confidence threshold）
        # 范围: 0.0 - 1.0
        # 含义: 只有检测到的人体边界框置信度 >= conf 时，才会被返回
        # 例如: conf=0.5 表示只返回置信度 >= 0.5 的检测结果
        # 值越小，检测越宽松（可能包含更多误检）
        # 值越大，检测越严格（只返回高置信度的结果）
        results = self.model(image_bgr, conf=0.5)

        if len(results[0].boxes) == 0:
            return None
        
        # 取第一张图片的所有识别姿态
        # keypoints.data 形状: [num_persons, 17, 3]
        # 第三维格式: [x坐标, y坐标, 置信度]
        keypoints = results[0].keypoints.data.cpu().numpy()
        
        # 取一张图片第一个检测到的人体
        # best_keypoints 形状: [17, 3] - 17个关键点，每个关键点是 [x, y, confidence]
        # 重要：无论关键点是否可见，模型都会返回所有17个关键点的数据
        # 不可见的关键点：置信度接近0，但坐标和置信度值仍然存在
        best_keypoints = keypoints[0].tolist()

        # 检查是否只有下半身
        # best_keypoints[index] 返回格式: [x, y, confidence] - 长度为3的列表
        # 例如: [123.45, 456.78, 0.95] 表示 x=123.45, y=456.78, 置信度=0.95
        left_knee = best_keypoints[self.KEYPOINTS["left_knee"]]      # [x, y, confidence]
        right_knee = best_keypoints[self.KEYPOINTS["right_knee"]]    # [x, y, confidence]
        left_ankle = best_keypoints[self.KEYPOINTS["left_ankle"]]    # [x, y, confidence]
        right_ankle = best_keypoints[self.KEYPOINTS["right_ankle"]]  # [x, y, confidence]
        
    
        if left_knee[2] > knee_conf or right_knee[2] > knee_conf:
            boolean=True
            return image_bgr, best_keypoints, boolean
        elif left_ankle[2] > ankle_conf or right_ankle[2] > ankle_conf:
            boolean=True
            return image_bgr, best_keypoints, boolean
        else:
            boolean=False
            return image_bgr, best_keypoints, boolean

    def check_half_person(self,image_path,knee_conf=0.5,ankle_conf=0.8):
        image=Image.open(image_path)
        image=np.array(image)
        # RGB转BGR
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        return self.check_half_person_from_array(image, knee_conf, ankle_conf)

    def draw_pose(self, image, keypoints_list, confidence_threshold=0.5, line_thickness=4, point_radius=5):
        """
        在图像上绘制姿态图
        
        Args:
            image: 原始图像 (BGR格式)
            keypoints_list: 关键点列表，格式为 [17, 3]，每个关键点是 [x, y, confidence]
            confidence_threshold: 置信度阈值，低于此值的关键点不绘制
            line_thickness: 连接线粗细
            point_radius: 关键点圆圈半径
            
        Returns:
            绘制了姿态的图像
        """
        # 复制图像，避免修改原图
        pose_image = image.copy()
        
        # 将关键点转换为numpy数组便于处理
        if isinstance(keypoints_list, list):
            keypoints = np.array(keypoints_list)
        else:
            keypoints = keypoints_list
        
        # 绘制连接线（骨架）
        for connection in self.SKELETON:
            pt1_idx, pt2_idx = connection
            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]
            
            # 只有当两个关键点的置信度都超过阈值时才绘制连接线
            if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                pt1_coord = (int(pt1[0]), int(pt1[1]))
                pt2_coord = (int(pt2[0]), int(pt2[1]))
                cv2.line(pose_image, pt1_coord, pt2_coord, self.LINE_COLOR, line_thickness)
        
        # 绘制关键点
        for i, kp in enumerate(keypoints):
            x, y, conf = kp
            if conf > confidence_threshold:
                center = (int(x), int(y))
                color = self.KP_COLORS[i]
                # 绘制实心圆
                cv2.circle(pose_image, center, point_radius, color, -1)
                # 绘制外圈（更明显）
                cv2.circle(pose_image, center, point_radius + 2, (255, 255, 255), 1)
        
        return pose_image
    

class YoloHalfBodyCheckNode:
    CATEGORY="zdx/logic"
    RETURN_TYPES=("BOOLEAN","IMAGE",)
    RETURN_NAMES=("boolean","pose_image",)
    FUNCTION="check_half"
    
    def __init__(self):
        # 初始化检测器（只加载一次模型）
        self.checker = YoloBodyPoseCheck()
    @classmethod
    def INPUT_TYPES(cls):
        return{
            "required":{
                "image":("IMAGE",),
            },
            "optional":{
                "knee_conf":("FLOAT",{"default":0.5,"min":0.0,"max":1.0,"step":0.01,}),
                "ankle_conf":("FLOAT",{"default":0.7,"min":0.0,"max":1.0,"step":0.01,}),
                "draw_conf":("FLOAT",{"default":0.5,"min":0.0,"max":1.0,"step":0.01,}),
                "line_thickness":("INT",{"default":4,"min":1,"max":10,"step":1,}),
                "point_radius":("INT",{"default":5,"min":1,"max":10,"step":1,}),
            }
        }

    def check_half(self, image, knee_conf, ankle_conf, draw_conf, line_thickness, point_radius):
        # 将tensor转换为PIL Image
        pil_image = tensor2pil(image)
        # 转换为numpy数组，然后转BGR格式
        image_np = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # 检测姿态（传入BGR格式的numpy数组）
        result = self.checker.check_half_person_from_array(image_bgr, knee_conf, ankle_conf)
        if result is None:
            # 没有检测到人体，返回原图和False
            return (False, image,)
        
        result_img, result_keypoints, result_boolean = result
        
        # 绘制姿态
        pose_image = self.checker.draw_pose(result_img, result_keypoints, draw_conf, line_thickness, point_radius)
        
        # 将BGR转回RGB，然后转换为tensor
        pose_image_rgb = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
        pose_image_pil = Image.fromarray(pose_image_rgb)
        pose_image_tensor = pil2tensor(pose_image_pil)
        
        return (result_boolean, pose_image_tensor,)

NODE_CLASS_MAPPINGS = {
    "yolo_detect": YoloDetect,
    "main_obj_extract": MainObjExtract,
    "YoloHalfBodyCheckNode": YoloHalfBodyCheckNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "yolo_detect": "yolo_detect",
    "main_obj_extract": "main_obj_extract",
    "YoloHalfBodyCheckNode": "YoloHalfBodyCheckNode",
}
# test:
if __name__ == '__main__':
    img = '/data/zdx/img/tryon_v1pose_fill_v3/32732942_refiner_pro.jpeg'
    img = '/data/zdx/img/tryon_case_omnitry_lora_v1/32662819_refiner_pro_debug.jpeg'
    out_debug = '/data/zdx/img/girl_half_hand_yolo_debug.jpeg'

    image = Image.open(img)
    w,h = image.size
    # resize到整除32
    image = image.resize((w//32*32, h//32*32))
    # 缩放到最长边为1536
    if max(w,h) > 1536:
        image = image.resize((w*1536//max(w,h), h*1536//max(w,h)))


    res = yolo_detect(
        pil2tensor(image), 
        confidence=0.5,
        detec_type = 'hand', debug=True
    )
    res['debug_image'].save(out_debug)

    # -- using class instance
    yolo_detect_node = YoloDetect()
    res = yolo_detect_node.forward(
        image=pil2tensor(image),
        yolo_model='hand_yolov8n.pt',
        confidence=0.5,
        debug=False
    )

