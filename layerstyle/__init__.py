from .image_scale_by_aspect_ratio_v2 import ImageScaleByAspectRatioV2


from .crop_by_mask import CropByMask
from .crop_by_mask_v2 import CropByMaskV2
from .crop_by_mask_v2 import CropByBBox
from .restore_crop_box import RestoreCropBox, RestoreCropBoxPad

from .image_mask_scale_as import LS_ImageMaskScaleAsV2, ImageMaskScaleAs
from .image_scale_by_aspect_ratio_v2 import ImageScaleByAspectRatioV2
from .image_scale_restore import ImageScaleRestore
from .image_scale_restore_v2 import ImageScaleRestoreV2
from .crop_box_resolve  import CropBoxResolve
from .image_blend_advance_v2 import ImageBlendAdvanceV2
from .color_image_v2 import ColorImageV2
from .yolov8_detect import YoloV8Detect

# bac  replace if no zdx_comfyui nodes~

# ImageScaleByAspectRatioV2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleByAspectRatio V2"]
# CropByMask = NODE_CLASS_MAPPINGS["LayerUtility: CropByMask"]
# CropByMaskV2 = NODE_CLASS_MAPPINGS["LayerUtility: CropByMask V2"]
# RestoreCropBox = NODE_CLASS_MAPPINGS["LayerUtility: RestoreCropBox"]
# ImageScaleRestoreV2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleRestore V2"]
# ImageScaleRestore = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleRestore"]
# CropBoxResolve = NODE_CLASS_MAPPINGS["LayerUtility: CropBoxResolve"]
# layerutility_cropbymask_v2 = NODE_CLASS_MAPPINGS["LayerUtility: CropByMask V2"]()
# layerutility_restorecropbox = NODE_CLASS_MAPPINGS["LayerUtility: RestoreCropBox"]()
# layerutility_imagescalebyaspectratio_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleByAspectRatio V2"]()
# layerutility_imagescalerestore_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleRestore V2"]()
# layerutility_cropbymask = NODE_CLASS_MAPPINGS["LayerUtility: CropByMask"]()
# layerutility_cropboxresolve = NODE_CLASS_MAPPINGS["LayerUtility: CropBoxResolve"]()
# layerutility_imageblendadvance_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageBlendAdvance V2"]()
# layerutility_colorimage_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ColorImage V2"]()


NODE_CLASS_MAPPINGS = {

    "ImageScaleByAspectRatioV2": ImageScaleByAspectRatioV2,
    "ImageScaleRestoreV2":ImageScaleRestoreV2,
    "ImageScaleRestore":ImageScaleRestore,

    "CropByMask": CropByMask,
    "CropByMaskV2":CropByMaskV2,
    "CropByBBox":CropByBBox,

    "CropBoxResolve": CropBoxResolve,
    "RestoreCropBox": RestoreCropBox,
    "RestoreCropBoxPad": RestoreCropBoxPad,

    "ImageMaskScaleAs": ImageMaskScaleAs,
    "ImageMaskScaleAsV2": LS_ImageMaskScaleAsV2,

    "ImageBlendAdvanceV2": ImageBlendAdvanceV2,
    "ColorImageV2": ColorImageV2,   
    "YoloV8Detect": YoloV8Detect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageScaleByAspectRatioV2": "ImageScaleByAspectRatioV2",

    "ImageScaleRestoreV2": "ImageScaleRestoreV2",
    "ImageScaleRestore": "ImageScaleRestore",

    "CropByMask": "CropByMask",
    "CropByMaskV2": "CropByMaskV2",
    "CropByBBox": "CropByBBox",
    "CropBoxResolve": "CropBoxResolve",
    "RestoreCropBox": "RestoreCropBox",
    "RestoreCropBoxPad": "RestoreCropBoxPad",

    "ImageMaskScaleAs": "ImageMaskScaleAs",
    "ImageMaskScaleAsV2": "ImageMaskScaleAsV2",
    
    "ImageBlendAdvanceV2": "ImageBlendAdvanceV2",
    "ColorImageV2": "ColorImageV2",

    "zdxYoloV8Detect": "YoloV8Detect",
}
