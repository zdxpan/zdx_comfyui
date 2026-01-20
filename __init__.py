__version__ = "3.0.0"

try:
    print(">> zdx_comfyui: initializing...")
    from .nodes import NODE_CLASS_MAPPINGS as Local_Mappings, NODE_DISPLAY_NAME_MAPPINGS as Local_Display_Mappings
    
    NODE_CLASS_MAPPINGS = Local_Mappings.copy()
    NODE_DISPLAY_NAME_MAPPINGS = Local_Display_Mappings.copy()

    # Dynamic Code
    try:
        from .dynamic_code import _NODE_CLASS_MAPPINGS as DynamicCodeMappings
        from .dynamic_code import _NODE_DISPLAY_NAME_MAPPINGS as DynamicCode_DISPLAY_MAPPING
        NODE_CLASS_MAPPINGS.update(DynamicCodeMappings)
        NODE_DISPLAY_NAME_MAPPINGS.update(DynamicCode_DISPLAY_MAPPING)
    except Exception as e:
        print(f"## zdx_comfyui dynamic_code load failed: {e}")

    # Sub-modules (refiner, supir, etc.)
    # Note: These modules seem to export classes but not a unified mapping in their __init__.
    # The original code imported classes directly in nodes.py and added them to dict.
    # We should reconstruct that logic here or in their respective modules.
    # Given the constraint to not touch too many files, we will import classes and update dict here.
    
    try:
        from .refiner_v1 import Refinerv1
        from .ic_light_v1 import ICLightv1
        from .refiner_v1pro import Refinerv1Pro
        from .refiner_f1 import RefinerF1
        from .supir import SupirXL
        from .focus_crop_match import FocusCrop, FocusCropV2, FocusCropUltra, DynamicAspectRatio, EditeMatch, FocusCropRestore, FocusCropRestoreUltra
        from .yolo_node import MainObjDetect, YoloDetect, MainObjExtract, YoloHalfBodyCheckNode
        from .repaint_f1 import HandFixF1
        from .repaint_v1 import RepaintV1
        from .layerstyle.seamless_blend import SeamlessBlend
        from .layerstyle.poisson_blend import PositionImageBlend

        sub_modules_mappings = {
            "QwenVlm": None, # Will be loaded later if available
            "MainObjDetect": MainObjDetect,
            "MainObjExtract": MainObjExtract,
            "YoloDetect": YoloDetect,
            "YoloHalfBodyCheckNode": YoloHalfBodyCheckNode,
            "SeamlessBlend": SeamlessBlend,
            "PositionImageBlend": PositionImageBlend,
            "FocusCrop": FocusCrop,
            "FocusCropV2": FocusCropV2,
            "FocusCropUltra": FocusCropUltra,
            "FocusCropRestore": FocusCropRestore,
            "FocusCropRestoreUltra": FocusCropRestoreUltra,
            "DynamicAspectRatio": DynamicAspectRatio,
            "EditeMatch": EditeMatch,
            "Refinerv1": Refinerv1,
            "ICLightv1": ICLightv1,
            "RepaintV1": RepaintV1,
            "Refinerv1Pro": Refinerv1Pro,
            "SupirXL": SupirXL,
            "RefinerF1": RefinerF1,
            "HandFixF1": HandFixF1,
        }
        
        sub_modules_display_names = {
             "QwenVlm": "QwenVlm",
             "MainObjDetect": "MainObjDetect",
             "MainObjExtract": "MainObjExtract",
             "zdxYoloDetect": "YoloDetect",
             "YoloHalfBodyCheckNode": "YoloHalfBodyCheckNode",
             "FocusCrop": "FocusCrop",
             "FocusCropV2": "FocusCropV2",
             "FocusCropUltra": "FocusCropUltra",
             "FocusCropRestore": "FocusCropRestore",
             "FocusCropRestoreUltra": "FocusCropRestoreUltra",
             "DynamicAspectRatio": "DynamicAspectRatio",
             "EditeMatch": "EditeMatch",
             "SeamlessBlend": "SeamlessBlend",
             "PositionImageBlend": "PositionImageBlend",
             "SizeNormalizer": "image size normalizer", # Already in local
             "zdxRefinerV1": "Refinerv1",
             "zdxICLightv1": "ICLightv1",
             "zdxRepaintV1": "RepaintV1",
             "zdxRefinerv1Pro": "Refinerv1Pro",
             "zdxSupirXL": "SupirXL",
             "zdxRefinerF1": "RefinerF1",
             "zdxHandFixF1": "HandFixF1",
        }
        
        # Filter None values
        NODE_CLASS_MAPPINGS.update({k: v for k, v in sub_modules_mappings.items() if v is not None})
        NODE_DISPLAY_NAME_MAPPINGS.update(sub_modules_display_names)
        
    except Exception as e:
        print(f"## zdx_comfyui sub-modules load failed: {e}")

    # LLM
    try:
        from .llm import QwenClient
        from .llm import NODE_CLASS_MAPPINGS as LLM_NODE_CLASS_MAPPINGS
        from .llm import NODE_DISPLAY_NAME_MAPPINGS as LLM_NODE_DISPLAY_NAME_MAPPING
        NODE_CLASS_MAPPINGS["QwenVlm"] = QwenClient
        NODE_CLASS_MAPPINGS.update(LLM_NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(LLM_NODE_DISPLAY_NAME_MAPPING)
    except Exception as e:
        print(f"## zdx_comfyui LLM load failed: {e}")

    # InstantFaceSwap
    try:
        from .face_reactor_instantid import InstantFaceSwap
        if InstantFaceSwap is not None:
            NODE_CLASS_MAPPINGS["InstantFaceSwap"] = InstantFaceSwap
            NODE_CLASS_MAPPINGS["XL InstantFaceSwap"] = InstantFaceSwap
            NODE_DISPLAY_NAME_MAPPINGS["XL InstantFaceSwap"] = "InstantFaceSwap"
    except Exception as e:
        print(f"## zdx_comfyui InstantFaceSwap load failed: {e}")

    # LayerStyle
    try:
        from .layerstyle import NODE_CLASS_MAPPINGS as layerstyle_NODE_CLASS_MAPPINGS
        from .layerstyle import NODE_DISPLAY_NAME_MAPPINGS as layerstyle_NODE_DISPLAY_NAME_MAPPINGS
        NODE_CLASS_MAPPINGS.update(layerstyle_NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(layerstyle_NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
         print(f"## zdx_comfyui LayerStyle load failed: {e}")

    # Mask
    try:
        from .mask import _NODE_CLASS_MAPPINGS as Mask_NODE_CLASS_MAPPINGS
        from .mask import _NODE_DISPLAY_NAME_MAPPINGS as Mask_NODE_DISPLAY_NAME_MAPPINGS
        NODE_CLASS_MAPPINGS.update(Mask_NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(Mask_NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
        print(f"## zdx_comfyui Mask load failed: {e}")

    # Annotator
    try:
        from .annotator import _NODE_CLASS_MAPPINGS as Annotator_NODE_CLASS_MAPPINGS
        from .annotator import _NODE_DISPLAY_NAME_MAPPINGS as Annotator_NODE_DISPLAY_NAME_MAPPINGS
        NODE_CLASS_MAPPINGS.update(Annotator_NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(Annotator_NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
        print(f"## zdx_comfyui Annotator load failed: {e}")

    # RMBG
    try:
        from .rmbg import NODE_CLASS_MAPPINGS as RMBG_NODE_CLASS_MAPPINGS
        from .rmbg import NODE_DISPLAY_NAME_MAPPINGS as RMBG_NODE_DISPLAY_NAME_MAPPINGS
        NODE_CLASS_MAPPINGS.update(RMBG_NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(RMBG_NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
        print(f"## zdx_comfyui RMBG load failed: {e}")

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
    print(f'\033[34m[ZDX ComfyUI]\033[0m v\033[93m{__version__}\033[0m | \033[93m{len(NODE_CLASS_MAPPINGS)} nodes\033[0m \033[92mLoaded\033[0m')

except Exception as e:
    print(f"## zdx_comfyui init failed: {e}")
    pass
