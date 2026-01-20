# must run at custom_nodes after custom nodes loaded!
import random
from typing import Sequence, Mapping, Any, Union
import torch
from .focus_crop_match import FocusCrop, EditeMatch

from nodes import NODE_CLASS_MAPPINGS


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]





class HandFixF1():
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "image": ("IMAGE",),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                        "denoise": ("FLOAT", {"default": 0.1,"min": 0.0, "max": 0.9, "step": 0.01}),
                    },
                    "optional" : {
                        "prompt": ("STRING", {"multiline": True}),
                        "mask": ("MASK",),
                        "model": ("MODEL",),
                        "clip": ("CLIP",),
                        "vae": ("VAE",),
                        # "reference_image": ("IMAGE",),
                        # "reference_mask": ("MASK",),
                    },
                }
    CATEGORY = "zdx/sampling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "forward"
    DESCRIPTION = """
repaint image for hand fix
"""
    def __init__(self, ):
        self.load = False

    @torch.inference_mode()
    def initiate(self):
        layermask_loadsam2model = NODE_CLASS_MAPPINGS["LayerMask: LoadSAM2Model"]()
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        self.dualcliploader_ = dualcliploader.load_clip(
            clip_name1="t5xxl_fp8_e4m3fn.safetensors",
            clip_name2="clip_l.safetensors",
            type="flux",
            device="default",
        )
        self.vaeloader_ = vaeloader.load_vae(vae_name="ae.safetensors")
        self.unetloader_dev_ = unetloader.load_unet(
            unet_name="flux1-dev-fp8.safetensors", weight_dtype="fp8_e4m3fn_fast"
        )
        self.sam2model_tiny_ = layermask_loadsam2model.load_sam2_model(
            sam2_model="sam2.1_hiera_tiny.safetensors", precision="fp16", device="cuda"
        )
        self.load = True

    @torch.inference_mode()
    def forward(self, image, model=None, clip=None, vae=None, seed=-1, denoise=0.5, prompt=None, mask=None):
        if not isinstance(image, tuple) and isinstance(image, torch.Tensor):
            image = (image, )
        
        layermask_objectdetectoryolo8 = NODE_CLASS_MAPPINGS["LayerMask: ObjectDetectorYOLO8"]()
        layermask_sam2ultrav2 = NODE_CLASS_MAPPINGS["LayerMask: SAM2UltraV2"]()
        layerutility_imagescalebyaspectratio_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleByAspectRatio V2"]()
        layermask_maskgrow = NODE_CLASS_MAPPINGS["LayerMask: MaskGrow"]()
        layerutility_imagescalerestore = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleRestore"]()
        layerutility_restorecropbox = NODE_CLASS_MAPPINGS["LayerUtility: RestoreCropBox"]()
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        focuscrop = FocusCrop()
        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        setlatentnoisemask = NODE_CLASS_MAPPINGS["SetLatentNoiseMask"]()
        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

        if model is None or clip is None or vae is None:
            self.initiate()
            model = self.unetloader_dev_
            clip =  self.dualcliploader_
            vae = self.vaeloader_
        if not isinstance(model, tuple):
            model = (model, )
        if not isinstance(clip, tuple):
            clip = (clip, )
        if not isinstance(vae, tuple):
            vae = (vae, )

        loadimage_init_1 = image

# ------object detect-----
        layermask_objectdetectoryolo8_441 = (
            layermask_objectdetectoryolo8.object_detector_yolo8(
                yolo_model="hand_yolov8s.pt",
                sort_method="left_to_right",
                bbox_select="all",
                select_index="0,1",
                image=get_value_at_index(loadimage_init_1, 0),
            )
        )

        layermask_sam2ultrav2_440 = layermask_sam2ultrav2.sam2_ultra(
            bbox_select="all",
            select_index="0,",
            detail_method="VITMatte",
            detail_erode=6,
            detail_dilate=4,
            black_point=0.15,
            white_point=0.99,
            process_detail=True,
            max_megapixels=1,
            sam2_model=get_value_at_index(self.sam2model_tiny_, 0),
            image=get_value_at_index(loadimage_init_1, 0),
            bboxes=get_value_at_index(layermask_objectdetectoryolo8_441, 0),
        )

        focuscrop_442 = focuscrop.crop_by_mask_v2(
            up_keep=0.5000000000000001,
            down_keep=0.5000000000000001,
            right_keep=0.5000000000000001,
            left_keep=0.5000000000000001,
            is_focus=True,
            image=get_value_at_index(layermask_sam2ultrav2_440, 0),
            mask=get_value_at_index(layermask_sam2ultrav2_440, 1),
        )

        layerutility_imagescalebyaspectratio_v2_ = (
            layerutility_imagescalebyaspectratio_v2.image_scale_by_aspect_ratio(
                aspect_ratio="original",
                proportional_width=1,
                proportional_height=1,
                fit="fill",
                method="lanczos",
                round_to_multiple="8",
                scale_to_side="longest",
                scale_to_length=768,
                background_color="#000000",
                image=get_value_at_index(focuscrop_442, 0),
                mask=get_value_at_index(focuscrop_442, 1),
            )
        )

# -- inference------

        loraloadermodelonly_439 = loraloadermodelonly.load_lora_model_only(
            lora_name="FLUX.1-Turbo-Alpha.safetensors",
            strength_model=1,
            model=get_value_at_index(model, 0),
        )

        loraloadermodelonly_437 = loraloadermodelonly.load_lora_model_only(
            lora_name="flux/flux_hand_v2.safetensors",
            strength_model=1,
            model=get_value_at_index(loraloadermodelonly_439, 0),
        )

        cliptextencode_433 = cliptextencode.encode(
            text=prompt if prompt  else "perfect hands", clip=get_value_at_index(clip, 0)
        )

        vaeencode_435 = vaeencode.encode(
            pixels=get_value_at_index(layerutility_imagescalebyaspectratio_v2_, 0),
            vae=get_value_at_index(vae, 0),
        )

        conditioningzeroout_430 = conditioningzeroout.zero_out(
            conditioning=get_value_at_index(cliptextencode_433, 0)
        )

        layermask_maskgrow_447 = layermask_maskgrow.mask_grow(
            invert_mask=False,
            grow=20,
            blur=4,
            mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_, 1),
        )

        setlatentnoisemask_436 = setlatentnoisemask.set_mask(
            samples=get_value_at_index(vaeencode_435, 0),
            mask=get_value_at_index(layermask_maskgrow_447, 0),
        )

        fluxguidance_446 = fluxguidance.append(
            guidance=3.5, conditioning=get_value_at_index(cliptextencode_433, 0)
        )

        ksampler_445 = ksampler.sample(
            seed=random.randint(1, 2**64) if seed == -1 else seed,
            steps=12,
            cfg=1,
            sampler_name="euler",
            scheduler="kl_optimal",
            denoise=denoise,
            model=get_value_at_index(loraloadermodelonly_437, 0),
            positive=get_value_at_index(fluxguidance_446, 0),
            negative=get_value_at_index(conditioningzeroout_430, 0),
            latent_image=get_value_at_index(setlatentnoisemask_436, 0),
        )

        vaedecode_444 = vaedecode.decode(
            samples=get_value_at_index(ksampler_445, 0),
            vae=get_value_at_index(vae, 0),
        )

        layerutility_imagescalerestore_471 = (
            layerutility_imagescalerestore.image_scale_restore(
                scale=1,
                method="lanczos",
                scale_by_longest_side=False,
                longest_side=1024,
                image=get_value_at_index(vaedecode_444, 0),
                mask=get_value_at_index(layermask_maskgrow_447, 0),
                original_size=get_value_at_index(
                    layerutility_imagescalebyaspectratio_v2_, 2
                ),
            )
        )

        layerutility_restorecropbox_475 = (
            layerutility_restorecropbox.restore_crop_box(
                invert_mask=False,
                background_image=get_value_at_index(loadimage_init_1, 0),
                croped_image=get_value_at_index(
                    layerutility_imagescalerestore_471, 0
                ),
                crop_box=get_value_at_index(focuscrop_442, 2),
            )
        )

        return layerutility_restorecropbox_475
