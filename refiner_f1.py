# must run at custom_nodes after custom nodes loaded!
import random
import torch
from PIL import Image, ImageFilter
from typing import Sequence, Mapping, Any, Union
from nodes import NODE_CLASS_MAPPINGS

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]
    

class RefinerF1():
    def __init__(self):
        self.refiner = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "image": ("IMAGE",),
                        "scale": ("FLOAT", { "default": 0.6, "min": 0.5, "max": 2, "step": 0.1, }),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                        "denoise": ("FLOAT", {"default": 0.1,"min": 0.0, "max": 0.9, "step": 0.01}),
                        "model": ("MODEL",),
                        "vae": ("VAE",),
                        "style_model": ("STYLE_MODEL",),
                        "clip_vision": ("CLIP_VISION",),
                        "redux_condition": ("CONDITIONING",),
                    },
                    "optional" : {
                        "mask": ("MASK",),
                    },
                }
    CATEGORY = "zdx/sampling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "call"
    DESCRIPTION = """
refiner image use F1.dev refiner model
"""
    def call(self, image, scale, seed, model, vae, style_model, clip_vision, redux_condition=None, mask=None, denoise=0.1):
        refiner_res = self.forward(image=image, scale=scale, seed=seed, denoise=denoise, model=model, vae=vae, style_model=style_model, clip_vision=clip_vision, redux_condition=redux_condition, mask=mask)

        return refiner_res


    # def __init__(self):   # no any model ,depend on the caller`s provider
    @torch.inference_mode()        
    def __init__(self):
        self.name = self.__class__.__name__
        # focus enhance
        self.layerutility_cropbymask_v2 = NODE_CLASS_MAPPINGS["LayerUtility: CropByMask V2"]()
        self.layerutility_restorecropbox = NODE_CLASS_MAPPINGS["LayerUtility: RestoreCropBox"]()
        self.layerutility_imagescalebyaspectratio_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleByAspectRatio V2"]()
        self.layerutility_imagescalerestore_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleRestore V2"]()
        self.growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
        self.imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()

        # uoscake gan
        self.upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        self.upscalemodelloader_690 = self.upscalemodelloader.load_model(
            model_name="RealESRGAN_x2plus.pth"
        )
        self.imagescaleby = NODE_CLASS_MAPPINGS["ImageScaleBy"]()


        # sampling
        self.fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        self.vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        self.ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        self.reduxadvanced = NODE_CLASS_MAPPINGS["ReduxAdvanced"]()


    @torch.inference_mode()
    def forward(self, image, scale=0.6, denoise=0.1, seed=-1, model=None, vae=None, style_model=None, clip_vision=None, redux_condition=None, mask=None):
        # upscale use esrgan x2
        if not isinstance(image, tuple) and isinstance(image, torch.Tensor):
            image = (image, )

        if mask is not None:
            if not isinstance(mask, tuple) and isinstance(mask, torch.Tensor):
                mask = (mask, )
            image_mask_cropbymask_1 = self.layerutility_cropbymask_v2.crop_by_mask_v2(
                invert_mask=False,
                detect="mask_area",
                top_reserve=8,
                bottom_reserve=8,
                left_reserve=8,
                right_reserve=8,
                round_to_multiple="8",
                image=get_value_at_index(image, 0),
                mask=get_value_at_index(mask, 0),
            )
            _, crop_h, crop_w, _ = image_mask_cropbymask_1[0].shape
            keep_pix = int(max(crop_h * 0.2, crop_w * 0.2))
            image_mask_cropbymask_1 = self.layerutility_cropbymask_v2.crop_by_mask_v2(
                invert_mask=False,
                detect="mask_area",
                top_reserve=keep_pix,
                bottom_reserve=keep_pix,
                left_reserve=keep_pix,
                right_reserve=keep_pix,
                round_to_multiple="8",
                image=get_value_at_index(image, 0),
                mask=get_value_at_index(mask, 0),
            )
            _, crop_h, crop_w, _ = image_mask_cropbymask_1[0].shape
        local_image = image if mask is None else image_mask_cropbymask_1
        imageupscalewithmodel_695 = self.imageupscalewithmodel.upscale(
            upscale_model=get_value_at_index(self.upscalemodelloader_690, 0),
            image=get_value_at_index(local_image, 0),
        )
        imagescaleby_694 = self.imagescaleby.upscale(
            upscale_method="lanczos",
            scale_by=scale, #0.6000000000000001,
            image=get_value_at_index(imageupscalewithmodel_695, 0),
        )

        if 1:
            ## cond, 使用参考图特征~
            fluxguidance_477 = self.fluxguidance.append(
                guidance=50, conditioning=get_value_at_index(self.cliptextencode_rep, 0)
            )
            reduxadvanced_475 = self.reduxadvanced.apply_stylemodel(
                downsampling_factor=2,
                downsampling_function="area",
                mode="center crop (square)",
                weight=1,
                autocrop_margin=0.1,
                conditioning=get_value_at_index(fluxguidance_477, 0),
                style_model=get_value_at_index(self.stylemodelloader_, 0),
                clip_vision=get_value_at_index(self.clipvisionloader_, 0),
                image=get_value_at_index(image_mask_cropbymask_1, 0),
                mask=get_value_at_index(image_mask_cropbymask_1, 1),
            )
            # encode as latent
            vaeencode_458 = self.vaeencode.encode(
                pixels=get_value_at_index(imagescaleby_694, 0),
                vae=get_value_at_index(self.vaeloader_, 0),
            )
            ksampler_390 = self.ksampler.sample(
                seed=random.randint(1, 2**64) if seed == -1 else seed,
                steps=12,
                cfg=1,
                sampler_name="euler",
                scheduler="normal",
                denoise=denoise,
                model=get_value_at_index(self.unetloader_dev_, 0),
                positive=get_value_at_index(reduxadvanced_475, 0),
                negative=get_value_at_index(self.conditioningzeroout_cond2, 0),
                latent_image=get_value_at_index(vaeencode_458, 0),
            )
            self.cleangpu(self.unetloader_dev_)
            repaint_area_res = self.vaedecode.decode(
                samples=get_value_at_index(ksampler_390, 0),
                vae=get_value_at_index(self.vaeloader_, 0),
            )

        # restore size and crop
        if mask is not None:
            # decode_res_resize_back_1 = (
            #     self.layerutility_imagescalebyaspectratio_v2.image_scale_by_aspect_ratio(
            #         aspect_ratio="custom",
            #         proportional_width=crop_w,
            #         proportional_height=crop_h,
            #         fit="letterbox",
            #         method="lanczos",
            #         round_to_multiple="8",
            #         scale_to_side="height",
            #         scale_to_length=1152,
            #         background_color="#000000",
            #         image=get_value_at_index(repaint_area_res, 0),
            #     )
            # )
            decode_res_resize_back_1 = self.imagescale.upscale(
                upscale_method="nearest-exact",
                width=crop_w,
                height=crop_h,
                crop="disabled",
                image=get_value_at_index(repaint_area_res, 0),
            )
            growmaskwithblur_479 = self.growmaskwithblur.expand_mask(
                expand=20,
                incremental_expandrate=1,
                tapered_corners=True,
                flip_input=False,
                blur_radius=20,
                lerp_alpha=1,
                decay_factor=1,
                fill_holes=False,
                mask=get_value_at_index(image_mask_cropbymask_1, 1),
            )
            repaint_area_res = (
                self.layerutility_restorecropbox.restore_crop_box(
                    invert_mask=False,
                    background_image=get_value_at_index(image, 0),
                    croped_image=get_value_at_index(decode_res_resize_back_1, 0),
                    crop_box=get_value_at_index(image_mask_cropbymask_1, 2),
                    croped_mask=get_value_at_index(growmaskwithblur_479, 0),
                )
            )

        # use original struct return
        return repaint_area_res

