# must run at custom_nodes after custom nodes loaded!
import random
import torch
from PIL import Image, ImageFilter
from typing import Sequence, Mapping, Any, Union
from nodes import NODE_CLASS_MAPPINGS


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]
    

class ControlnetPoseCond():

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "init_image": ("IMAGE",),
                        "ref_image": ("IMAGE",),
                        "controlnet": ("CONTROL_NET",),
                        "cond1": ("CONDITIONING",),
                        "cond2": ("CONDITIONING",),
                        "vae": ("VAE",),
                    },
                }
    CATEGORY = "zdx/cond"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("cond",)
    FUNCTION = "flux_controlnet_cond"
    DESCRIPTION = """
flux openpose cond
"""
    def flux_controlnet_cond(self, init_image, ref_image, cond1, cond2, vae):
        setunioncontrolnettype = NODE_CLASS_MAPPINGS["SetUnionControlNetType"]()
        imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()
        aio_preprocessor = NODE_CLASS_MAPPINGS["AIO_Preprocessor"]()
        easy_imageconcat = NODE_CLASS_MAPPINGS["easy imageConcat"]()
        controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
        getimagesize = NODE_CLASS_MAPPINGS["GetImageSize"]()

        setunioncontrolnettype_98 = setunioncontrolnettype.set_controlnet_type(
            type="openpose", control_net=get_value_at_index(self.controlnetloader_, 0)
        )
        _, _h, _w, _ = init_image.shape
        
        emptyimagepro_777 = emptyimagepro.generate(
            batch_size=1, color=0, image=ref_image,
        )
        aio_preprocessor_pose = aio_preprocessor.execute(
            preprocessor="DWPreprocessor", resolution=512,
            image=init_image,
        )
        imagescale_345 = imagescale.upscale(
            upscale_method="lanczos",
            width=_w,
            height=_h,
            crop="disabled",
            image=get_value_at_index(aio_preprocessor_pose, 0),
        )
        self.cleangpu(aio_preprocessor_pose)
        pose_black_concat = easy_imageconcat.concat(
            direction="right",
            match_image_size=False,
            image1=get_value_at_index(emptyimagepro_777, 0),
            image2=get_value_at_index(imagescale_345, 0),
        )
        controlnetapplyadvanced_cond1 = controlnetapplyadvanced.apply_controlnet(
            strength=1.0,
            start_percent=0,
            end_percent=1,
            positive=cond1,
            negative=cond2,
            control_net=get_value_at_index(setunioncontrolnettype_98, 0),
            image=get_value_at_index(pose_black_concat, 0),
            vae=vae,
        )
        self.cleangpu(self.controlnetloader_)
        self.cleangpu(setunioncontrolnettype_98)
        return controlnetapplyadvanced_cond1

NODE_CLASS_MAPPINGS["ControlnetPoseCond"] = ControlnetPoseCond
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS["ControlnetPoseCond"] = "ControlnetPoseCond"
