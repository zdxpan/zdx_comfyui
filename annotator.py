import cv2
import copy
import numpy as np
import math, os, time
import sys
import torch
from collections import Counter
import mediapipe as mp
from insightface.app import FaceAnalysis

from PIL              import Image,ImageDraw,  ImageOps, __version__

GLOBAL_FACE_APP = None


def pil2tensor(image):
    new_image = image.convert('RGB')
    img_array = np.array(new_image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)[None]
    return img_tensor

def tensor2pil(image):
    if len(image.shape) < 3:
        image = image.unsqueeze(0)
    return Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))



def get_min_bounding_box(img):
    img = img.convert('L')
    img_array = np.array(img)

    # find all pixl is white area
    coords = np.argwhere(img_array > 200)

    if coords.size == 0:
        return (0, 0, img.width, img.height)

    # get the boundary of minimal outer rectangle
    left = coords[:, 1].min()
    top = coords[:, 0].min()
    right = coords[:, 1].max()
    bottom = coords[:, 0].max()

    return (left, top, right, bottom)

def insight_detect_face(insightface, image, only_detect=True):
    ''' image is nparray'''
    face = []
    recognition = None
    if only_detect:
        recognition = insightface.models.pop('recognition', None)
    try:
        for size in [(size, size) for size in range(640, 256, -64)]:
            insightface.det_model.input_size = size # TODO: hacky but seems to be working
            face = insightface.get(image)
            if face:
                break
        if len(face) > 0:
            face.sort(key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]), reverse = True) # 降序排序
    finally:
        if only_detect and recognition is not None:
            insightface.models['recognition'] = recognition
    return face


def get_mediapipe_model_path(model_type="segmenter"):
    try:
        import folder_paths
        model_folder_path = os.path.join(folder_paths.models_dir, "mediapipe")
    except Exception:
        model_folder_path = os.path.join("~/.mediapipe")
    
    if model_type == "segmenter":
        model_name = "selfie_multiclass_256x256.tflite"
        model_url = f"https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/{model_name}"
    else:
        model_name = "face_landmarker.task"
        model_url = f"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float32/latest/{model_name}"
    
    model_file_path = os.path.join(model_folder_path, model_name)
    
    if not os.path.exists(model_file_path):
        os.makedirs(model_folder_path, exist_ok=True)
        try:
            import wget
            print(f"Downloading '{model_name}' model to {model_file_path}")
            wget.download(model_url, model_file_path)
        except Exception:
            import urllib.request
            print(f"Downloading '{model_name}' via urllib...")
            urllib.request.urlretrieve(model_url, model_file_path)
    return model_file_path


class MediaPipeDetector:
    def __init__(self):
        # 1. Initialize Segmentation Model (for high-quality mask)
        seg_model_path = get_mediapipe_model_path("segmenter")
        with open(seg_model_path, "rb") as f:
            seg_buffer = f.read()
        self.seg_options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=seg_buffer),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(self.seg_options)

        # 2. Initialize 468-point Face Mesh Model (for precise BBox)
        mesh_model_path = get_mediapipe_model_path("face_mesh")
        with open(mesh_model_path, "rb") as f:
            mesh_buffer = f.read()
        self.mesh_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=mesh_buffer),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(self.mesh_options)

    def get_face_data(self, image_np):
        """
        Get 468 landmarks, BBox, and semantic segmentation mask
        """
        h, w, _ = image_np.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        
        # Run Mesh and Segmenter
        mesh_result = self.landmarker.detect(mp_image)
        seg_result = self.segmenter.segment(mp_image)
        
        # Extract semantic segmentation face skin mask (Category 3)
        face_skin_mask = seg_result.confidence_masks[3].numpy_view()
        
        m_x1, m_y1, m_x2, m_y2 = float('inf'), float('inf'), float('-inf'), float('-inf')
        mesh_coords = None
        if mesh_result.face_landmarks:
            landmarks = mesh_result.face_landmarks[0]
            mesh_coords = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
            m_x1, m_y1 = mesh_coords.min(axis=0)
            m_x2, m_y2 = mesh_coords.max(axis=0)

        s_x1, s_y1, s_x2, s_y2 = float('inf'), float('inf'), float('-inf'), float('-inf')
        # Use a slightly lower threshold for bbox calculation to ensure coverage
        if np.any(face_skin_mask > 0.1):
            y_coords, x_coords = np.where(face_skin_mask > 0.1)
            s_x1, s_y1, s_x2, s_y2 = x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()
        
        # Union of mesh bbox and segmentation bbox
        x1 = min(m_x1, s_x1)
        y1 = min(m_y1, s_y1)
        x2 = max(m_x2, s_x2)
        y2 = max(m_y2, s_y2)

        if x1 == float('inf'):
            # Return empty/zeros if no face detected
            return None, None, face_skin_mask
        
        # Return [x1, y1, x2, y2]
        bbox = np.array([x1, y1, x2, y2])
        return mesh_coords, bbox, face_skin_mask

    def close(self):
        if hasattr(self, 'segmenter'): self.segmenter.close()
        if hasattr(self, 'landmarker'): self.landmarker.close()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            "optional": {
                "max_faces": ("INT", {"default": 1, "min": 1, "max": 10}), # Future support
            }
        }
    
    CATEGORY = "zdx/face"
    RETURN_TYPES = ("MASK", "BBOX", "MESH_COORDS")
    RETURN_NAMES = ("face_skin_mask", "bbox", "mesh_coords")
    FUNCTION = "detect"
    DESCRIPTION = """
    Detect face and return skin mask, bbox, and mesh coordinates. Supports batch processing.
    """

    def detect(self, input_image, max_faces=1):
        # input_image: (B, H, W, C) tensor
        batch_size, h, w, _ = input_image.shape
        
        mask_list = []
        bbox_list = []
        mesh_list = []

        for i in range(batch_size):
            # Convert tensor to numpy (H, W, C) uint8
            img_np = (input_image[i].cpu().numpy() * 255).astype(np.uint8)
            
            mesh_coords, bbox, face_skin_mask = self.get_face_data(img_np)
            
            if mesh_coords is None:
                # No face found: use empty mask, zero bbox, zero mesh
                mask_list.append(torch.zeros((h, w), dtype=torch.float32))
                bbox_list.append(torch.zeros((4,), dtype=torch.float32))
                # 468 points, 2 coords
                mesh_list.append(torch.zeros((468, 2), dtype=torch.float32))
            else:
                mask_list.append(torch.from_numpy(face_skin_mask).float())
                bbox_list.append(torch.from_numpy(bbox).float())
                mesh_list.append(torch.from_numpy(mesh_coords).float())

        # Stack results
        # Mask: (B, H, W)
        masks = torch.stack(mask_list)
        # BBox: (B, 4)
        bboxes = torch.stack(bbox_list)
        # Mesh: (B, 468, 2)
        meshes = torch.stack(mesh_list)

        return (masks, bboxes, meshes)

class ManualFaceAnalysis:
    # 'bbox', 'kps', # not 'landmark_3d_68', 'pose', 'landmark_2d_106', provider

    def __init__(self, model_path, providers, name='buffalo_l'):
        self.models = {}
        # Default detection model for buffalo_l
        # Try finding the model in the expected subdirectory first
        det_model_path = os.path.join(model_path, 'models', name, 'det_10g.onnx')
        if not os.path.exists(det_model_path):
             # Try direct path just in case
             det_model_path = os.path.join(model_path, 'det_10g.onnx')
        
        if not os.path.exists(det_model_path):
             raise FileNotFoundError(f"Detection model not found at {det_model_path}")
        from insightface.model_zoo import get_model
        print(f"InsightFace: Manually loading detection model from {det_model_path}...")
        self.det_model = get_model(det_model_path, providers=providers)
        self.det_model.prepare(ctx_id=0, input_size=(640, 640))
        
    def get(self, img):
        bboxes, kpss = self.det_model.detect(img, max_num=0, metric='default')
        if bboxes is None:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = dict(bbox=bbox, kps=kps, det_score=det_score)
            ret.append(face)
        return ret

class BaseDetector():
    def __init__(self, model_path, device='cpu'):
        self.model = None
        self.model_path = model_path
        self.device = device

    @torch.no_grad()
    def __call__(self, image, **kwargs):
        if self.model == None:
            self.load()
        self.to_device()
        try:
            return self.forward(image, **kwargs)
        finally:
            self.to_cpu()

    def forward(self, image, detect_resolution=512, image_resolution=512):
        return self.model(
            image,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution
        )

    def load(self):
        raise Exception('Not implemented')

    def to_cpu(self):
        self.model.to('cpu')

    def to_device(self):
        self.model.to(self.device)


class FaceDetector(BaseDetector):
    def __init__(self, model_path=None, device='cpu'):
        global GLOBAL_FACE_APP
        
        if model_path is None:
            try:
                import folder_paths
                model_path = folder_paths.get_full_path('insightface')
            except Exception:
                model_path = os.path.expanduser("~/.insightface")
             
        self.model_path = model_path
        self.device = device
        
        self._init_mediapipe()
        self._init_insightface(GLOBAL_FACE_APP)

    def _init_mediapipe(self):
        self.use_new_api = False
        self.face_landmarker = None
        self.segmenter = None
        self.face_mesh = None
        self.face_detection = None
        
        try:
            import mediapipe.tasks.python as mp_python
            from mediapipe.tasks.python import vision
            
            # Initialize Face Landmarker
            mesh_model_path = get_mediapipe_model_path("face_mesh")
            with open(mesh_model_path, "rb") as f:
                mesh_buffer = f.read()
            base_options = mp_python.BaseOptions(model_asset_buffer=mesh_buffer)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options, 
                output_face_blendshapes=False, 
                output_facial_transformation_matrixes=False, 
                num_faces=1)
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

            # Initialize Image Segmenter (for better face mask)
            seg_model_path = get_mediapipe_model_path("segmenter")
            with open(seg_model_path, "rb") as f:
                seg_buffer = f.read()
            seg_options = vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_buffer=seg_buffer),
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True
            )
            self.segmenter = vision.ImageSegmenter.create_from_options(seg_options)

            self.use_new_api = True
        except Exception as e:
            print(f"InsightFace: Failed to initialize MediaPipe New API: {e}")
            self.use_new_api = False
            
        if not self.use_new_api:
            try:
                self.face_mesh = mp.solutions.face_mesh
                # self.face_detection = mp.solutions.face_detection
            except Exception:
                print("InsightFace: Failed to initialize MediaPipe face mesh and detection.")
                self.face_mesh = None
                # self.face_detection = None
    
    def _init_insightface(self, global_face_app):
        if global_face_app is not None:
            self.FACE_APP = global_face_app
            return
            
        try:
            start_time = time.time()
            print(f"InsightFace: Start loading models from {self.model_path} at {time.strftime('%X')}...")
            
            try:
                self.FACE_APP = ManualFaceAnalysis(self.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], name='buffalo_l')
                print(f"InsightFace: Manual loading successful.")
            except Exception as e:
                print(f"InsightFace: Manual loading failed ({e}), falling back to FaceAnalysis...")
                self.FACE_APP = FaceAnalysis(name='buffalo_l', root=self.model_path,
                                                allowed_modules=['detection'],
                                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                
                ctx_id = 0
                try:
                    if self.FACE_APP.models:
                        first_model = next(iter(self.FACE_APP.models.values()))
                        if hasattr(first_model, 'session'):
                            if 'CUDAExecutionProvider' not in first_model.session.get_providers():
                                ctx_id = -1
                                print("InsightFace: CUDA provider not active, falling back to CPU (ctx_id=-1)")
                except Exception:
                    pass

                self.FACE_APP.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            global GLOBAL_FACE_APP
            GLOBAL_FACE_APP = self.FACE_APP
            print(f"InsightFace: Models loaded successfully in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"InsightFace initialization failed: for FaceDetectir {e}")
            self.FACE_APP = None

    @classmethod
    def INPUT_TYPES(cls):
        fit_mode = ["all", 'crop']
        return {
                    "required": {
                        "input_image": ("IMAGE",),
                        "fit": (fit_mode, "all"),
                        "expand_rate": ("FLOAT", { "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.05, }),
                        "only_one": ("BOOLEAN", { "default": True, }),
                        "invert": ("BOOLEAN", { "default": False, }),
                        "use_insight_mesh": ("BOOLEAN", { "default": False, }),
                    }
                }
    CATEGORY = "zdx/face"
    RETURN_TYPES = ("IMAGE", "MASK", "BOX",)
    RETURN_NAMES = ("image", "mask", "original_size",)
    FUNCTION = "call"
    DESCRIPTION = """
get face mask or face area
"""
    def call(self, input_image, fit, expand_rate=0.5, only_one=False, invert=False, use_insight_mesh=False):
        # 1. Convert Tensor [B, H, W, C] to Numpy [B, H, W, C] (uint8) once
        # This eliminates the repeated tensor2pil conversion in the loop
        input_image_np = (input_image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        results_image = []
        results_mask = []
        results_box = []
        
        batch_size = input_image_np.shape[0]
        
        for i in range(batch_size):
            img_np = input_image_np[i] # [H, W, C]
            
            if fit != 'crop':
                # Forward now accepts and returns numpy
                mask_np = self.forward(img_np, expand_rate=expand_rate, only_one=only_one, invert=invert, use_insight_mesh=use_insight_mesh)
                
                # Convert back to tensor
                # Image: Reuse input tensor slice to save memory/time
                results_image.append(input_image[i].unsqueeze(0))
                
                # Mask: [H, W] -> [1, H, W] (float32, 0-1)
                mask_tensor = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0)
                results_mask.append(mask_tensor)
                results_box.append(None)
            else:
                # Process crop mode using numpy
                out_image_np, mask_np, box = self._process_crop(img_np, expand_rate, use_insight_mesh=use_insight_mesh)
                
                # Convert crop result to tensor
                img_tensor = torch.from_numpy(out_image_np.astype(np.float32) / 255.0).unsqueeze(0)
                results_image.append(img_tensor)
                
                if mask_np is not None:
                     mask_tensor = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0)
                     results_mask.append(mask_tensor)
                else:
                    # Empty mask matching crop size
                    h, w = out_image_np.shape[:2]
                    empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
                    results_mask.append(empty_mask)
                    
                results_box.append(box)

        # Batch results
        if fit != 'crop':
            return (torch.cat(results_image, dim=0), torch.cat(results_mask, dim=0), None)
        else:
            try:
                final_image = torch.cat(results_image, dim=0)
                final_mask = torch.cat(results_mask, dim=0)
                return (final_image, final_mask, results_box)
            except RuntimeError:
                print(f"Warning: Batch crop sizes mismatch. Returning first image only.")
                return (results_image[0], results_mask[0], results_box[0])

    def _process_crop(self, img_np, expand_rate=0.5, use_insight_mesh=False):
        # img_np is [H, W, 3] uint8
        faces_infos = self.detect_face_area(img_np, expand_rate=expand_rate)
        
        if len(faces_infos) == 0:
            return (img_np, None, None)
            
        face_info = faces_infos[0]
        box = face_info['box']
        # Numpy slicing for crop: [y1:y2, x1:x2]
        crop_img_np = img_np[box[1]:box[3], box[0]:box[2]]
        
        mask_np = None
        # Always generate mask for crop mode if requested (logic from original call_pil)
        # Note: Original logic implied fit='mask' or 'all', but call passes fit parameter check outside.
        # Here we just generate it.
        
        face_kps = None
        if 'kps' in face_info:
            kps = face_info['kps']
            # Adjust KPS to crop coordinates
            face_kps = kps - np.array([box[0], box[1]])

        mask_np = self.face_mask(crop_img_np, kps=face_kps, use_insight_mesh=use_insight_mesh)
            
        return (crop_img_np, mask_np, box)

    def forward(self, image_np, **kwargs):
        # Expects numpy array [H, W, 3]
        if not hasattr(self, 'FACE_APP'):
            self.load()
        
        expand_rate = kwargs.get("expand_rate", 0.5)
        invert = kwargs.get("invert", True)
        use_insight_mesh = kwargs.get("use_insight_mesh", False)
        
        # Detect faces (detect_face_area accepts numpy)
        faces = self.detect_face_area(image_np, expand_rate)
        
        # Prepare Mask [H, W] - 0: Black, 255: White
        # invert=True: White Background (255), Face will be Black (0)
        # invert=False: Black Background (0), Face will be White (255)
        bg_color = 255 if invert else 0
        mask_np = np.full(image_np.shape[:2], bg_color, dtype=np.uint8)
        
        if not faces:
            return mask_np
            
        for face in faces:
            bbox = face['box']
            # Crop using slicing
            face_crop_np = image_np[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            if face_crop_np.size == 0: continue

            # face_mask returns: Black Face (0), White BG (255) (uint8 numpy)
            
            # Adjust keypoints to crop coordinates
            face_kps = None
            if 'kps' in face:
                kps = face['kps']
                face_kps = kps - np.array([bbox[0], bbox[1]])

            mask_crop_np = self.face_mask(face_crop_np, face_kps, use_insight_mesh=use_insight_mesh)
            
            # Logic mapping:
            # Mask Crop is: 0 (Face), 255 (BG)
            
            if not invert:
                # We want: White Face (255), Black BG (0)
                # Input: Black Face (0), White BG (255)
                # So we invert the crop mask: 255-0=255(Face), 255-255=0(BG)
                mask_crop_np = 255 - mask_crop_np
            else:
                # We want: Black Face (0), White BG (255)
                # Input is already: Black Face (0), White BG (255)
                pass
                
            # Paste the mask back
            # We only want to paste the 'Face' part.
            # If !invert (Face=255, BG=0): Paste non-zero pixels? 
            # Actually, standard paste replaces the rectangle.
            # But the mask_crop_np has "BG" pixels too.
            # However, in the original code, it just pasted the rectangular crop.
            # Let's replicate that behavior first.
            
            mask_np[bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask_crop_np
            
            if kwargs.get("only_one", True):
                break

        return mask_np
    
    def _draw_mask_from_landmarks(self, shape, landmarks_list):
        h, w = shape[:2]
        # Default White BG (255)
        blank_image = np.full((h, w), 255, dtype=np.uint8)
        
        for landmarks in landmarks_list:
            # landmarks is list of (x, y) normalized
            points = np.array([[int(l[0] * w), int(l[1] * h)] for l in landmarks], dtype=np.int32)
            hull = cv2.convexHull(points)
            cv2.fillPoly(blank_image, [hull], 0) # Fill Black
            
        return blank_image # Return Numpy directly

    def face_mask(self, image, kps=None, use_insight_mesh=False):
        # image is numpy array (H, W, 3) in RGB format
        h, w = image.shape[:2]

        # 0. Try Segmenter (Best Quality) - New API
        if not use_insight_mesh and self.use_new_api and self.segmenter:
             try:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                
                # Run Segmenter
                seg_result = self.segmenter.segment(mp_image)
                # Category 3 is Face Skin
                face_skin_mask = seg_result.confidence_masks[3].numpy_view()
                
                # We want: Face=0 (Black), BG=255 (White)
                # face_skin_mask is 0-1 probability of being face.
                mask = np.full((h, w), 255, dtype=np.uint8)
                # If confidence > 0.1, it's face (0)
                mask[face_skin_mask > 0.1] = 0
                
                return mask
             except Exception as e:
                print(f">> Face Segmenter processing failed: {e}")

        # 1. Try New API (FaceLandmarker) - Fallback to landmarks if segmenter fails
        if not use_insight_mesh and self.use_new_api and self.face_landmarker:
            try:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                detection_result = self.face_landmarker.detect(mp_image)
                
                if detection_result.face_landmarks:
                    landmarks_list = [[(l.x, l.y) for l in face] for face in detection_result.face_landmarks]
                    return self._draw_mask_from_landmarks((h, w), landmarks_list)
            except Exception as e:
                print(f">> FaceLandmarker processing failed: {e}")
        
        # 2. Try Legacy API (FaceMesh)
        if not use_insight_mesh and self.face_mesh is None and not self.use_new_api:
             try:
                self.face_mesh = mp.solutions.face_mesh
                self.face_detection = mp.solutions.face_detection
             except Exception:
                pass # Continue to fallback

        if not use_insight_mesh and self.face_mesh:
            try:
                with self.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5) as face_mesh_:
                    results = face_mesh_.process(image)
                    
                if results and results.multi_face_landmarks:
                    landmarks_list = [[(l.x, l.y) for l in face.landmark] for face in results.multi_face_landmarks]
                    return self._draw_mask_from_landmarks((h, w), landmarks_list)
            except Exception as e:
                print(f">> FaceMesh processing failed: {e}")
        
        # 3. Fallback: Use InsightFace 5 keypoints (if provided)
        if kps is not None and len(kps) > 0:
            # kps contains: Left Eye, Right Eye, Nose, Left Mouth, Right Mouth
            # We calculate the center and expand the points to cover the face
            center = np.mean(kps, axis=0)
            # Expand factor to cover forehead and chin roughly
            # 1.8 to 2.0 usually covers the whole face area from the feature points
            expanded_pts = []
            for p in kps:
                vec = p - center
                expanded_pts.append(center + vec * 2.2) 
            
            # Add forehead point (approximate)
            # Midpoint of eyes + up vector
            left_eye = kps[0]
            right_eye = kps[1]
            eye_center = (left_eye + right_eye) / 2
            nose = kps[2]
            # Vector from nose to eye center roughly points up
            up_vec = eye_center - nose
            forehead = eye_center + up_vec * 1.5
            expanded_pts.append(forehead)
            
            expanded_pts = np.array(expanded_pts, dtype=np.int32)
            hull = cv2.convexHull(expanded_pts)
            
            # Default White BG (255)
            blank_image = np.full((h, w), 255, dtype=np.uint8)
            cv2.fillPoly(blank_image, [hull], 0) # Fill Black
            return blank_image

        # 4. Fallback: Simple Ellipse (assuming centered face)
        # Default White BG (255)
        blank_image = np.full((h, w), 255, dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (w // 2, h // 2)
        cv2.ellipse(blank_image, center, axes, 0, 0, 360, 0, -1) # Fill Black
        
        return blank_image
    
    def detect_face_area(self, image, expand_rate=0.05):
        """detect face using InsightFace, input as cv2 image rgb """
        if not hasattr(self, 'FACE_APP'):
            self.load()
        face_info = insight_detect_face(self.FACE_APP, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        if len(face_info) < 1:
            return []
        # res = ['bbox', 'kps', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'embedding']
        height, width, _ = image.shape
        for face_ in face_info:
            face_['box'] = face_['bbox']
            face_['box_old'] = face_['box']
            
            # Extract landmarks for alignment (Triangle: Left Eye, Right Eye, Mouth Center)
            if 'kps' in face_:
                kps = face_['kps']
                if len(kps) >= 5:
                    # InsightFace KPS: 0=LeftEye, 1=RightEye, 2=Nose, 3=LeftMouth, 4=RightMouth
                    face_['left_eye'] = kps[0]
                    face_['right_eye'] = kps[1]
                    # Calculate mouth center from corners
                    face_['mouth_center'] = (kps[3] + kps[4]) / 2.0
                    face_['triangle'] = np.array([face_['left_eye'], face_['right_eye'], face_['mouth_center']])

            bbox = face_['bbox'].astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            face_w = right - left
            face_h = bottom - top
            center_x = left + face_w // 2
            center_y = top + face_h // 2
            face_['width'] = face_w
            face_['height'] = face_h
            face_['center'] = (center_x, center_y)
            face_['area'] = face_w * face_h
            # Calculate expanded dimensions
            face_w_dt = int(face_w * expand_rate)
            face_h_dt = int(face_h * expand_rate)
            # Calculate new box based on center
            face_['box'] = (
                max(0, center_x - face_w // 2 - face_w_dt),
                max(0, center_y - face_h // 2 - face_h_dt),
                min(width, center_x + face_w // 2 + face_w_dt),
                min(height, center_y + face_h // 2 + face_h_dt)
            )
            face_['box'] = tuple(map(int, face_['box']))
            
        return face_info

def is_empty_image(image):
    if image is None:
        return True
    if image.mode == 'RGBA':
        extrema = image.getextrema()
        if extrema[3][1] == 0:
            return True
    return False


class InsightFaceCrop:
    """
    crop face  and get face eye and mouth center triangle,
    """
    def __init__(self):
        self.face_masker = FaceDetector(model_path = None)

    @classmethod
    def INPUT_TYPES(cls):
        fit_mode = ["all", 'crop']
        return {
                    "required": {
                        "input_image": ("IMAGE",),
                        "expand_rate": ("FLOAT", { "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.05, }),
                        "index": ("INT", { "default": 0, "min": 0, "max": 4, "step": 1, }),
                    }
                }
    CATEGORY = "zdx/face"
    RETURN_TYPES = ("IMAGE", "MASK", "TRIANGLE", "BOX",)
    RETURN_NAMES = ("image", "mask", "triangle", "box",)
    FUNCTION = "crop"
    DESCRIPTION = """
crop face area from image, and get triangle of face eye and mouth center to align face
"""
    def crop(self, input_image, expand_rate=0.5, index=0):
        image = tensor2pil(input_image)
        face_info2 = self.face_masker.detect_face_area(np.array(image), expand_rate=expand_rate)
        
        if len(face_info2) == 0:
            return (input_image, np.zeros((3,2)), [0,0,0,0])
            
        # sort face_info2 by area   
        face_info2 = sorted(face_info2, key=lambda x: x['width'] * x['height'], reverse=True)
        
        if index >= len(face_info2):
            index = 0
            
        selected_face = face_info2[index]
        box = selected_face['box']
        
        face_crop_ = image.crop(box)
        mask_empty = np.zeros((face_crop_.height, face_crop_.width), dtype=np.uint8)
        # TODO insightface  if it`s has mask to indecate face area?
        # Adjust triangle coordinates to be relative to the crop
        triangle = selected_face['triangle'].copy()
        triangle[:, 0] -= box[0]
        triangle[:, 1] -= box[1]
        
        return (pil2tensor(face_crop_), mask_empty, triangle, list(box))

class FaceAlignScale:
    """
    align face by triangle, and scale face to target size
    通过inisight_face 检测人脸，如果想将两个大小角度不一样的人脸进行缩放到相同大小并对其的话（比如贴上去）调整透明度，算法思路是啥
    将目标人脸缩放到与源人脸相同的大小和角度（只缩放旋转，不扭曲），然后替换。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "source_triangle": ("TRIANGLE",),
                "target_triangle": ("TRIANGLE",),
            }
        }
    
    CATEGORY = "zdx/face"
    RETURN_TYPES = ("IMAGE", "TRIANGLE", "BOX",)
    RETURN_NAMES = ("aligned_image", "aligned_triangle", "box",)
    FUNCTION = "align"
    
    def align(self, source_image, target_image, source_triangle, target_triangle):
        # source_image: tensor [1, H, W, C]
        # source_triangle: numpy array [[x1,y1], [x2,y2], [x3,y3]] (left_eye, right_eye, mouth_center)
        src_img_pil = tensor2pil(source_image)
        tgt_img_pil = tensor2pil(target_image)
        # 1. Calculate centers of triangles
        src_center = np.mean(source_triangle, axis=0)
        tgt_center = np.mean(target_triangle, axis=0)
        # 2. Calculate scale (based on eye distance)
        # Eye distance is distance between point 0 and 1
        src_eye_dist = np.linalg.norm(source_triangle[0] - source_triangle[1])
        tgt_eye_dist = np.linalg.norm(target_triangle[0] - target_triangle[1])
        scale = tgt_eye_dist / src_eye_dist
        # 3. Calculate rotation angle
        # Vector between eyes
        src_eye_vec = source_triangle[1] - source_triangle[0]
        tgt_eye_vec = target_triangle[1] - target_triangle[0]
        src_angle = np.degrees(np.arctan2(src_eye_vec[1], src_eye_vec[0]))
        tgt_angle = np.degrees(np.arctan2(tgt_eye_vec[1], tgt_eye_vec[0]))
        rotation_angle = tgt_angle - src_angle
        
        # 4. Perform affine transformation
        # Translation to center -> Rotate & Scale -> Translation to new center
        # Since we want to align src_center to tgt_center
        
        # We can use OpenCV's getRotationMatrix2D for rotation and scaling around a center
        M = cv2.getRotationMatrix2D((float(src_center[0]), float(src_center[1])), float(rotation_angle), float(scale))
        
        # Adjust translation part of the matrix to align centers
        # Current transformation maps src_center to itself (because we rotated around it)
        # We need to add translation (tgt_center - src_center)
        M[0, 2] += (tgt_center[0] - src_center[0])
        M[1, 2] += (tgt_center[1] - src_center[1])
        
        w, h = src_img_pil.size
        # Use target image size for output canvas
        tgt_w, tgt_h = tgt_img_pil.size
        
        aligned_img_np = cv2.warpAffine(
            np.array(src_img_pil), 
            M, 
            (tgt_w, tgt_h), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0,0,0)
        )
        # Transform the source triangle to new coordinates for verification/downstream use
        ones = np.ones(shape=(len(source_triangle), 1))
        points_ones = np.hstack([source_triangle, ones])
        aligned_triangle = M.dot(points_ones.T).T
        return (pil2tensor(Image.fromarray(aligned_img_np)), aligned_triangle, None)


_NODE_CLASS_MAPPINGS = {
    "FaceAlignScale": FaceAlignScale,
    'FaceDetector': FaceDetector,
    'InsightFaceCrop': InsightFaceCrop,
    "MediaPipeDetector": MediaPipeDetector,

}
_NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceAlignScale": "FaceAlignScale",
    "FaceDetector": "FaceDetector",
    "InsightFaceCrop": "InsightFaceCrop",
    "MediaPipeDetector": "MediaPipeDetector",
}

if __name__ == '__main__':
    from workflows.zdx_comfyui.annotator import FaceDetector
    insight_face_path = '/data/models/insightface/'
    imgs = ['/data/zdx/gallery/5_final_res.png', '/data/zdx/gallery/0018_after.jpeg']

    face_masker = FaceDetector(model_path = insight_face_path)
    # insight_detect_face(face_masker.FACE_APP, cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
    for im in imgs:
        im = Image.open(im)
        
        face_info2 = face_masker.detect_face_area(np.array(im), None, expand_rate=0.5)
        face_crop_ = im.crop(face_info2[0]['box'])
        face_crop_.save('1__face_crop.png')
        # face_info = face_masker.detect_face_area_v2(np.array(im), face_masker.face_detection)
            
        face_mask = face_masker.forward(image = im, expand_rate=0.5, invert=False)
        face_mask.save('1__face_mask.png')
        new_im = Image.new('RGBA', im.size, 255)
        new_im.paste(im=im, mask=face_mask.convert('L'))
        # new_im.putalpha(face_mask.convert('L'))
        # face_mask.save('1__face_mask.png')
        # new_im.save('1__face_mask.png')

    # ----人脸对齐测试gu----------
    aliner =  FaceAlignScale()
    image1 = imgs[0]
    image2 = imgs[1]
    source_image = Image.open(image1)
    target_image = Image.open(image2)
    face_info_source = face_masker.detect_face_area(np.array(source_image), None, expand_rate=0.5)
    face_info_target = face_masker.detect_face_area(np.array(target_image), None, expand_rate=0.5)
    if len(face_info_source) == 0 or len(face_info_target) == 0:
        raise ValueError("No face detected in one of the images")
    source_triangle = face_info_source[0]['triangle']
    target_triangle = face_info_target[0]['triangle']
    aligned_image, aligned_triangle, box = aliner.align(pil2tensor(source_image), pil2tensor(target_image), source_triangle, target_triangle)
    source_image_face_crop_ = source_image.crop(face_info_source[0].box)
    source_image_face_crop_.save('1__source_image_face_crop.png')
    target_image_face_crop_ = target_image.crop(face_info_target[0].box)
    target_image_face_crop_.save('1__target_image_face_crop.png')
    tensor2pil(aligned_image).save('1__aligned_image.png')

