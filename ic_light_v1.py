import os
import random
import sys
import json
import argparse
from typing import Sequence, Mapping, Any, Union
import torch

from nodes import NODE_CLASS_MAPPINGS


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]



class ICLightv1():
    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "image": ("IMAGE",),
                        "prompt": ("STRING", {"multiline": True}),
                        "ipscale": ("FLOAT", { "default": 0.6, "min": 0.5, "max": 2, "step": 0.1, }),
                        "icmultiple": ("FLOAT", { "default": 0.182, "min": 0.01, "max": 0.9, "step": 0.01, }),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                        "denoise": ("FLOAT", {"default": 0.7,"min": 0.0, "max": 0.9, "step": 0.01}),
                    },
                    "optional" : {
                        "reference_image": ("IMAGE",),
                        "negative_prompt": ("STRING", { "default": "low quality", "multiline": True}),
                        "model": ("MODEL",),
                        "clip": ("CLIP",),
                        "vae": ("VAE",),
                    },
                }
    CATEGORY = "zdx/sampling"
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("IMAGE", "ICIMAGE",)
    FUNCTION = "call"
    DESCRIPTION = """
iclight  v1.5
"""        
    def call(self, image, prompt, ipscale, icmultiple, seed, denoise, reference_image=None, negative_prompt=None, model=None, clip=None, vae=None):
        ic_res = self.forward(image=image,prompt=prompt, ipscale=ipscale, icmultiple=icmultiple,
             seed=seed, denoise=denoise, reference_image=reference_image, negative_prompt=negative_prompt, model=model, clip=clip, vae=vae)  
        return ic_res

    @torch.inference_mode()
    def initiate(self,):
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        clipsetlastlayer = NODE_CLASS_MAPPINGS["CLIPSetLastLayer"]()
        easy_stylesselector = NODE_CLASS_MAPPINGS["easy stylesSelector"]()
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        
        self.checkpointloadersimple_ = checkpointloadersimple.load_checkpoint(
            ckpt_name="juggernaut_reborn.safetensors"
        )
        self.clipsetlastlayer_47 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-2, clip=get_value_at_index(self.checkpointloadersimple_, 1)
        )
        easy_stylesselector_75 = easy_stylesselector.run(
            styles="fooocus_styles",
            positive="",
            negative="low quality",
        )
        self.cond1 = cliptextencode.encode(
            text=get_value_at_index(easy_stylesselector_75, 0),
            clip=get_value_at_index(self.clipsetlastlayer_47, 0),
        )
        self.cond2 = cliptextencode.encode(
            text=get_value_at_index(easy_stylesselector_75, 1),
            clip=get_value_at_index(self.clipsetlastlayer_47, 0),
        )

    @torch.inference_mode()
    def forward(self, image, prompt='', ipscale=0.5, icmultiple=0.182, seed=-1, denoise=0.7, reference_image=None, negative_prompt="", model=None, clip=None, vae=None):
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        easy_positive = NODE_CLASS_MAPPINGS["easy positive"]()
        easy_negative = NODE_CLASS_MAPPINGS["easy negative"]()
        clipsetlastlayer = NODE_CLASS_MAPPINGS["CLIPSetLastLayer"]()
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        layerutility_imagescalebyaspectratio_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleByAspectRatio V2"]()
        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        iclightconditioning = NODE_CLASS_MAPPINGS["ICLightConditioning"]()
        ipadapterunifiedloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()
        ipadapteradvanced = NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]()
        freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()
        loadandapplyiclightunet = NODE_CLASS_MAPPINGS["LoadAndApplyICLightUnet"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        layerutility_imageblend = NODE_CLASS_MAPPINGS["LayerUtility: ImageBlend"]()
        detailtransfer = NODE_CLASS_MAPPINGS["DetailTransfer"]()
        imagecasharpening = NODE_CLASS_MAPPINGS["ImageCASharpening+"]()
        imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()

        # prompt = "Sunlight, Dindar light efficiency"
        prompt = prompt.strip()
        negative_prompt="low quality"
        
        if not model or not clip or not vae:
            self.initiate()
            self.model = 1
            model = self.checkpointloadersimple_[0]
            clip = get_value_at_index(self.clipsetlastlayer_47, 0)
            vae = get_value_at_index(self.checkpointloadersimple_, 2)
        # loadimage_init = loadimage.load_image(image="20250822-115124.jpeg")
        if not isinstance(image, tuple) and isinstance(image, torch.Tensor):
            image = (image, )
        loadimage_init = image
        _, image_h, image_w, _ = loadimage_init[0].shape

        if not hasattr(self, 'cond1'):
            easy_stylesselector_75 = easy_stylesselector.run(
                styles="fooocus_styles",
                positive=prompt,
                negative="low quality",
            )
            self.cond1 = cliptextencode.encode(
                text=get_value_at_index(easy_stylesselector_75, 0),
                clip=clip,
            )
            self.cond2 = cliptextencode.encode(
                text=get_value_at_index(easy_stylesselector_75, 1),
                clip=clip,
            )

        cliptextencode_57 = self.cond1
        cliptextencode_58 = self.cond2
        if prompt:
            cliptextencode_57 = cliptextencode.encode(
                text=prompt,
                clip=clip,
            )
        if negative_prompt:
            cliptextencode_58 = cliptextencode.encode(
                text=negative_prompt,
                clip=clip,
            )

        layerutility_imagescalebyaspectratio_v2_23 = (
            layerutility_imagescalebyaspectratio_v2.image_scale_by_aspect_ratio(
                aspect_ratio="original",
                proportional_width=1,
                proportional_height=1,
                fit="crop",
                method="lanczos",
                round_to_multiple="8",
                scale_to_side="longest",
                scale_to_length=1152,
                background_color="#000000",
                image=get_value_at_index(loadimage_init, 0),
            )
        )
        vaeencode_72 = vaeencode.encode(
            pixels=get_value_at_index(layerutility_imagescalebyaspectratio_v2_23, 0),
            vae=vae,
        )
        iclightconditioning_79 = iclightconditioning.encode(
            multiplier=icmultiple,    # multiplier=0.182,
            positive=get_value_at_index(cliptextencode_57, 0),
            negative=get_value_at_index(cliptextencode_58, 0),
            vae=vae,
            foreground=get_value_at_index(vaeencode_72, 0),
        )

        ipadapterunifiedloader_55 = ipadapterunifiedloader.load_models(
            preset="PLUS (high strength)",
            model=model,
        )
        ipadapteradvanced_54 = ipadapteradvanced.apply_ipadapter(
            weight=ipscale,
            weight_type="linear",
            combine_embeds="concat",
            start_at=0,
            end_at=0.5,
            embeds_scaling="V only",
            model=get_value_at_index(ipadapterunifiedloader_55, 0),
            ipadapter=get_value_at_index(ipadapterunifiedloader_55, 1),
            image=get_value_at_index(layerutility_imagescalebyaspectratio_v2_23, 0),
        )
        freeu_v2_64 = freeu_v2.patch(
            b1=1.5,
            b2=1.6,
            s1=0.9,
            s2=0.2,
            model=get_value_at_index(ipadapteradvanced_54, 0),
        )
        loadandapplyiclightunet_65 = loadandapplyiclightunet.load(
            model_path="IC-Light/iclight_sd15_fc.safetensors",
            model=get_value_at_index(freeu_v2_64, 0),
        )
        ksampler_66 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=30,
            cfg=3,
            sampler_name="dpmpp_2m_sde",
            scheduler="karras",
            denoise=denoise,
            model=get_value_at_index(loadandapplyiclightunet_65, 0),
            positive=get_value_at_index(iclightconditioning_79, 0),
            negative=get_value_at_index(iclightconditioning_79, 1),
            latent_image=get_value_at_index(iclightconditioning_79, 2),
        )
        vaedecode_52 = vaedecode.decode(
            samples=get_value_at_index(ksampler_66, 0),
            vae=vae,
        )
        decode_res_resize_back_1 = imagescale.upscale(
            upscale_method="nearest-exact",
            width=image_w,
            height=image_h,
            crop="disabled",
            image=get_value_at_index(vaedecode_52, 0),
        )
        layerutility_imageblend_70 = layerutility_imageblend.image_blend(
            invert_mask=False,
            blend_mode="screen",
            opacity=50,
            background_image=get_value_at_index(decode_res_resize_back_1, 0),
            layer_image=get_value_at_index(loadimage_init, 0),
        )

        detailtransfer_71 = detailtransfer.process(
            mode="add",
            blur_sigma=1,
            blend_factor=1,
            target=get_value_at_index(layerutility_imageblend_70, 0),
            source=get_value_at_index(
                layerutility_imagescalebyaspectratio_v2_23, 0
            ),
        )

        imagecasharpening_80 = imagecasharpening.execute(
            amount=0.15000000000000002,
            image=get_value_at_index(detailtransfer_71, 0),
        )
        return (imagecasharpening_80[0], vaedecode_52[0])

