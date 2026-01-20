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
    

class SupirXL():
    def __init__(self):
        self.refiner = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "image": ("IMAGE",),
                        "scale": ("FLOAT", { "default": 0.6, "min": 0.5, "max": 2, "step": 0.1, }),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                    },
                    "optional" : {
                        "prompt": ("STRING", {"multiline": True}),
                        "mask": ("MASK",),
                        "upscale": ("BOOLEAN", {"default": False}),
                    },
                }
    CATEGORY = "zdx/sampling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "call"


    DESCRIPTION = """
    refiner image use supir
"""
    def call(self, image, scale, seed=-1, prompt=None, mask=None, upscale=False):
        if not self.refiner:
            self.initiate()
            self.refiner = 1
        refiner_res = self.forward(image, scale=scale, seed=seed,  prompt=prompt, mask=mask, upscale=upscale)
        return refiner_res

    @torch.inference_mode()
    def initiate(self):
        self.name = self.__class__.__name__

        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        supir_model_loader_v2 = NODE_CLASS_MAPPINGS["SUPIR_model_loader_v2"]()
        self.supir_first_stage = NODE_CLASS_MAPPINGS["SUPIR_first_stage"]()
        self.supir_encode = NODE_CLASS_MAPPINGS["SUPIR_encode"]()
        self.supir_conditioner = NODE_CLASS_MAPPINGS["SUPIR_conditioner"]()
        self.supir_sample = NODE_CLASS_MAPPINGS["SUPIR_sample"]()
        self.supir_decode = NODE_CLASS_MAPPINGS["SUPIR_decode"]()
        self.imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()        

        self.imagescaleby = NODE_CLASS_MAPPINGS["ImageScaleBy"]()
        self.upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()

        self.upscalemodelloader_ = self.upscalemodelloader.load_model(
            model_name="RealESRGAN_x2plus.pth"
        )
        checkpointloadersimple_ = checkpointloadersimple.load_checkpoint(
            ckpt_name="Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors"
        )
        self.supir_model_loader_v2_ = supir_model_loader_v2.process(
            supir_model="SUPIR-v0Q.fp16.safetensors",
            fp8_unet=False,
            diffusion_dtype="auto",
            high_vram=False,
            model=get_value_at_index(checkpointloadersimple_, 0),
            clip=get_value_at_index(checkpointloadersimple_, 1),
            vae=get_value_at_index(checkpointloadersimple_, 2),
        )

        # focus enhance
        self.layerutility_cropbymask_v2 = NODE_CLASS_MAPPINGS["LayerUtility: CropByMask V2"]()
        self.layerutility_restorecropbox = NODE_CLASS_MAPPINGS["LayerUtility: RestoreCropBox"]()
        self.layerutility_imagescalebyaspectratio_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleByAspectRatio V2"]()
        self.layerutility_imagescalerestore_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleRestore V2"]()
        self.growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
        self.imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
        self.imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()
        
        self.easy_cleangpuused = NODE_CLASS_MAPPINGS["easy cleanGpuUsed"]()
        self.gpu_models = [
            self.supir_model_loader_v2_,
        ]        


    @torch.inference_mode()
    def forward(self, image, scale=0.5, seed=-1, prompt=None, mask=None, upscale=False):
        if not self.refiner:
            self.initiate()
            self.refiner = 1

        prompt_res = prompt + "best quality, intricate details, sharp focus, fine details realistic texture" if prompt is not None else "best quality, intricate details, sharp focus, fine details realistic texture"
        if not isinstance(image, tuple) and isinstance(image, torch.Tensor):
            image = (image, )
        _, H, W, _ = image[0].shape
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
            upscale_model=get_value_at_index(self.upscalemodelloader_, 0),
            image=get_value_at_index(local_image, 0),
        )
        imagescaleby_28 = self.imagescaleby.upscale(
            upscale_method="lanczos",
            scale_by=scale,
            image=get_value_at_index(imageupscalewithmodel_695, 0),
        )

        supir_first_stage_5 = self.supir_first_stage.process(
            use_tiled_vae=True,
            encoder_tile_size=1024,
            decoder_tile_size=1024,
            encoder_dtype="auto",
            SUPIR_VAE=get_value_at_index(self.supir_model_loader_v2_, 1),
            image=get_value_at_index(imagescaleby_28, 0),
        )
        supir_encode_11 = self.supir_encode.encode(
            use_tiled_vae=True,
            encoder_tile_size=1024,
            encoder_dtype="auto",
            SUPIR_VAE=get_value_at_index(supir_first_stage_5, 0),
            image=get_value_at_index(supir_first_stage_5, 1),
        )

        supir_conditioner_9 = self.supir_conditioner.condition(
            positive_prompt= prompt_res,
            negative_prompt="bad quality, blurry, messy",
            SUPIR_model=get_value_at_index(self.supir_model_loader_v2_, 0),
            latents=get_value_at_index(supir_first_stage_5, 2),
        )

        supir_sample_7 = self.supir_sample.sample(
            seed=random.randint(1, 2**64) if seed == -1 else seed,
            steps=12,
            cfg_scale_start=4.0,
            cfg_scale_end=4.0,
            EDM_s_churn=5,
            s_noise=1.003,
            DPMPP_eta=1,
            control_scale_start=1.0,
            control_scale_end=1.0,
            restore_cfg=1,
            keep_model_loaded=False,
            sampler="RestoreDPMPP2MSampler",
            sampler_tile_size=1024,
            sampler_tile_stride=512,
            SUPIR_model=get_value_at_index(self.supir_model_loader_v2_, 0),
            latents=get_value_at_index(supir_encode_11, 0),
            positive=get_value_at_index(supir_conditioner_9, 0),
            negative=get_value_at_index(supir_conditioner_9, 1),
        )
        self.cleangpu((self.supir_model_loader_v2_[0], ))

        supir_decode_10 = self.supir_decode.decode(
            use_tiled_vae=True,
            decoder_tile_size=1024,
            SUPIR_VAE=get_value_at_index(self.supir_model_loader_v2_, 1),
            latents=get_value_at_index(supir_sample_7, 0),
        )
        self.cleangpu((self.supir_model_loader_v2_[1], ))
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
            #         image=get_value_at_index(supir_decode_10, 0),
            #     )
            # )
            decode_res_resize_back_1 = self.imagescale.upscale(
                upscale_method="nearest-exact",
                width=crop_w,
                height=crop_h,
                crop="disabled",
                image=get_value_at_index(supir_decode_10, 0),
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
            supir_decode_10 = (
                self.layerutility_restorecropbox.restore_crop_box(
                    invert_mask=False,
                    background_image=get_value_at_index(image, 0),
                    croped_image=get_value_at_index(decode_res_resize_back_1, 0),
                    crop_box=get_value_at_index(image_mask_cropbymask_1, 2),
                    croped_mask=get_value_at_index(growmaskwithblur_479, 0),
                )
            )
        if not upscale:
            supir_decode_10 = self.imageresizekj.resize(
                width=W,
                height=H,
                upscale_method="bicubic",
                keep_proportion=True,
                divisible_by=1,
                crop="disabled",
                image=get_value_at_index(supir_decode_10, 0),
            )
        return supir_decode_10

    def cleangpu(self, x):
        self.easy_cleangpuused.empty_cache(
                anything=get_value_at_index(x, 0),
                unique_id=None,
            )
