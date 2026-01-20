# must run at custom_nodes after custom nodes loaded!
import random
import torch
from PIL import Image, ImageFilter
from typing import Sequence, Mapping, Any, Union
from nodes import NODE_CLASS_MAPPINGS
from .layerstyle import (
    ImageScaleByAspectRatioV2,
    LS_ImageMaskScaleAsV2, ImageMaskScaleAs,
    ImageScaleRestore, ImageScaleRestoreV2,
    CropByMask,  CropByMaskV2, 
    CropBoxResolve,
    RestoreCropBox,
    ImageBlendAdvanceV2, ColorImageV2
)



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
    

class Refinerv1():
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
                    },
                    "optional" : {
                        "prompt": ("STRING", {"multiline": True}),
                        "mask": ("MASK",),
                        "reference_image": ("IMAGE",),
                        "reference_mask": ("MASK",),
                    },
                }
    CATEGORY = "zdx/sampling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "call"
    DESCRIPTION = """
refiner image use v1.5 tile controlnet
"""
    def call(self, image, scale,seed=-1, denoise=0.1, prompt=None,mask=None, reference_image=None, reference_mask=None):
        if not self.refiner:
            self.initiate()
            self.refiner = 1
        refiner_res = self.forward(image=image, scale=scale, seed=seed, denoise=denoise, prompt=prompt,mask=mask, reference_image=reference_image, reference_mask=reference_mask)

        return refiner_res


    # def __init__(self):
    @torch.inference_mode()        
    def initiate(self):
        self.name = self.__class__.__name__
        self.clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        self.imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
        self.imagescaleby = NODE_CLASS_MAPPINGS["ImageScaleBy"]()

        self.imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
        
        self.loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        self.vaeencodetiled = NODE_CLASS_MAPPINGS["VAEEncodeTiled"]()
        self.controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        self.controlnetloader_697 = self.controlnetloader.load_controlnet(
            control_net_name="control_v11f1e_sd15_tile.pth"
        )
        self.freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()
        self.perturbedattentionguidance = NODE_CLASS_MAPPINGS["PerturbedAttentionGuidance"]()
        self.automatic_cfg = NODE_CLASS_MAPPINGS["Automatic CFG"]()
        self.tileddiffusion = NODE_CLASS_MAPPINGS["TiledDiffusion"]()
        self.controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
        self.samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()
        self.vaedecodetiled = NODE_CLASS_MAPPINGS["VAEDecodeTiled"]()
        self.imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()
        # add ipadapter

        self.checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        self.checkpointloadersimple_685 = self.checkpointloadersimple.load_checkpoint(
            ckpt_name="juggernaut_reborn.safetensors"
        )
        loraloader_687 = self.loraloader.load_lora(
            lora_name="v15/more_details.safetensors",
            strength_model=0.2,
            strength_clip=0.2,
            model=get_value_at_index(self.checkpointloadersimple_685, 0),
            clip=get_value_at_index(self.checkpointloadersimple_685, 1),
        )

        self.loraloader_686 = self.loraloader.load_lora(
            lora_name="v15/SDXLrender_v2.0.safetensors",
            strength_model=0.1,
            strength_clip=0.2,
            model=get_value_at_index(loraloader_687, 0),
            clip=get_value_at_index(loraloader_687, 1),
        )
        self.freeu_v2_691 = self.freeu_v2.patch(
            b1=0.9,
            b2=1.08,
            s1=0.9500000000000001,
            s2=0.8,
            model=get_value_at_index(self.loraloader_686, 0),
        )
        if hasattr(self.perturbedattentionguidance, 'patch'):
            self.perturbedattentionguidance_675 = self.perturbedattentionguidance.patch(
                scale=1, model=get_value_at_index(self.freeu_v2_691, 0)
            )
        else:
            self.perturbedattentionguidance_675 = self.perturbedattentionguidance.execute(
                scale=1, model=get_value_at_index(self.freeu_v2_691, 0)
            )
        self.automatic_cfg_688 = self.automatic_cfg.patch(
            hard_mode=True,
            boost=True,
            model=get_value_at_index(self.perturbedattentionguidance_675, 0),
        )

        self.tileddiffusion_698 = self.tileddiffusion.apply(
            method="MultiDiffusion",
            tile_width=1024,
            tile_height=1024,
            tile_overlap=128,
            tile_batch_size=4,
            model=get_value_at_index(self.automatic_cfg_688, 0),
        )
        self.cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        self.cliptextencode_676 = self.cliptextencode.encode(
            text="(worst quality, low quality, normal quality:1.5)",
            clip=get_value_at_index(self.loraloader_686, 1),
        )
        self.cliptextencode_684 = self.cliptextencode.encode(
            text="masterpiece, best quality, highres",
            clip=get_value_at_index(self.loraloader_686, 1),
        )

        self.alignyourstepsscheduler = NODE_CLASS_MAPPINGS["AlignYourStepsScheduler"]()

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        self.ksamplerselect_678 = ksamplerselect.get_sampler(sampler_name="dpmpp_3m_sde_gpu")

        self.upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        self.upscalemodelloader_690 = self.upscalemodelloader.load_model(
            model_name="RealESRGAN_x2plus.pth"
        )
        self.imagescaleby = NODE_CLASS_MAPPINGS["ImageScaleBy"]()

        # focus enhance
        self.layerutility_cropbymask_v2 = CropByMaskV2()
        self.layerutility_restorecropbox = RestoreCropBox()
        self.layerutility_imagescalebyaspectratio_v2 = ImageScaleByAspectRatioV2()
        self.layerutility_imageblendadvance_v2 = ImageBlendAdvanceV2()
        self.layerutility_imagescalerestore_v2 = ImageScaleRestore()
        self.layerutility_cropboxresolve = CropBoxResolve()
        # self.layerutility_cropbymask_v2 = NODE_CLASS_MAPPINGS["LayerUtility: CropByMask V2"]()
        # self.layerutility_restorecropbox = NODE_CLASS_MAPPINGS["LayerUtility: RestoreCropBox"]()
        # self.layerutility_imagescalebyaspectratio_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleByAspectRatio V2"]()
        # self.layerutility_imagescalerestore_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleRestore V2"]()
        self.growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
        self.imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
            

    @torch.inference_mode()
    def forward(self, image, scale=0.6, denoise=0.1, seed=-1, prompt=None, mask=None, reference_image = None, reference_mask=None):
        # upscale use esrgan x2
        if not isinstance(image, tuple) and isinstance(image, torch.Tensor):
            image = (image, )
        if not self.refiner:
            self.initiate()
            self.refiner = 1
        cond1 = self.cliptextencode_684
        if prompt is not None and len(prompt) > 1:
            cond1 = self.cliptextencode.encode(
                text=prompt,
                clip=get_value_at_index(self.loraloader_686, 1),
            )
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
        # encode as latent
        vaeencodetiled_693 = self.vaeencodetiled.encode(
            tile_size=1024,
            overlap=64,
            temporal_size=64,
            temporal_overlap=8,
            pixels=get_value_at_index(imagescaleby_694, 0),
            vae=get_value_at_index(self.checkpointloadersimple_685, 2),
        )

        alignyourstepsscheduler = NODE_CLASS_MAPPINGS["AlignYourStepsScheduler"]()
        alignyourstepsscheduler_677 = alignyourstepsscheduler.get_sigmas(
            model_type="SD1", steps=30, denoise=denoise
        )
        controlnetapplyadvanced_699 = self.controlnetapplyadvanced.apply_controlnet(
            strength=1,
            start_percent=0.1,
            end_percent=1,
            positive=get_value_at_index(cond1, 0),
            negative=get_value_at_index(self.cliptextencode_676, 0),
            control_net=get_value_at_index(self.controlnetloader_697, 0),
            image=get_value_at_index(local_image, 0),
        )

#-------load and apply ipadapter0---------
        final_model = get_value_at_index(self.tileddiffusion_698, 0)
        dtype = torch.bfloat16
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float32 if device == 'cpu' else dtype
        with torch.autocast(device, dtype=dtype):
            samplercustom_689 = self.samplercustom.sample(
                add_noise=True,
                noise_seed=random.randint(1, 2**64) if seed == -1 else seed,
                cfg=8,
                model=final_model,
                positive=get_value_at_index(controlnetapplyadvanced_699, 0),
                negative=get_value_at_index(controlnetapplyadvanced_699, 1),
                sampler=get_value_at_index(self.ksamplerselect_678, 0),
                sigmas=get_value_at_index(alignyourstepsscheduler_677, 0),
                latent_image=get_value_at_index(vaeencodetiled_693, 0),
            )
            vaedecodetiled_679 = self.vaedecodetiled.decode(
                tile_size=1024,
                samples=get_value_at_index(samplercustom_689, 0),
                vae=get_value_at_index(self.checkpointloadersimple_685, 2),
                # vae=self.vaeloader_12[0],
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
            #         image=get_value_at_index(vaedecodetiled_679, 0),
            #     )
            # )
            decode_res_resize_back_1 = self.imagescale.upscale(
                upscale_method="nearest-exact",
                width=crop_w,
                height=crop_h,
                crop="disabled",
                image=get_value_at_index(vaedecodetiled_679, 0),
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
            vaedecodetiled_679 = (
                self.layerutility_restorecropbox.restore_crop_box(
                    invert_mask=False,
                    background_image=get_value_at_index(image, 0),
                    croped_image=get_value_at_index(decode_res_resize_back_1, 0),
                    crop_box=get_value_at_index(image_mask_cropbymask_1, 2),
                    croped_mask=get_value_at_index(growmaskwithblur_479, 0),
                )
            )

        # use original struct return
        return vaedecodetiled_679

