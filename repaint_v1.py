import os
import random
import sys
import json
import argparse
import contextlib
from typing import Sequence, Mapping, Any, Union
import torch


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




class RepaintV1():
    @torch.inference_mode()
    def __init__(self):
        self.checkpointloadersimple_ = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "reference": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True}),
                "reference_mask": ("MASK",),
                "controlnet_image": ("IMAGE",),
                "controlnet": ("CONTROL_NET",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
        }
    CATEGORY = "zdx/sampling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "forward"
    DESCRIPTION = """
repaint image for v1.5 model
"""
    @torch.inference_mode()
    def initiate(self):
        # 加载模型
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        self.checkpointloadersimple_ = checkpointloadersimple.load_checkpoint(
            ckpt_name="juggernaut_reborn.safetensors"
        )
        # controlnet_image = aio_preprocessor.execute(
        #     preprocessor="DWPreprocessor",
        #     resolution=512,
        #     image=get_value_at_index(layerutility_imagescalebyaspectratio_v2_43, 0),
        # )

    @torch.inference_mode()
    def forward(self, image, mask=None, reference=None, reference_mask=None, prompt='', model=None, clip=None, vae=None, 
            controlnet_image=None, controlnet=None, seed=-1, denoise=1.0, grow_mask=30):
        if not isinstance(image, tuple) and isinstance(image, torch.Tensor):
            image = (image,)
        if not isinstance(mask, tuple) and isinstance(mask, torch.Tensor):
            mask = (mask,)
        # loadimage_reference_2 = loadimage.load_image(image="03_1.jpg")
        # loadimage_init_1 = loadimage.load_image(image="27_1.jpg")
        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        # aio_preprocessor = NODE_CLASS_MAPPINGS["AIO_Preprocessor"]()
        conditioningaverage = NODE_CLASS_MAPPINGS["ConditioningAverage"]()
        controlnetapply = NODE_CLASS_MAPPINGS["ControlNetApply"]()
        ipadapterunifiedloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()
        ipadapteradvanced = NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]()
        freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()
        growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
        inpainteasymodel = NODE_CLASS_MAPPINGS["InpaintEasyModel"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        from .layerstyle import NODE_CLASS_MAPPINGS as layerstyle_NODE_CLASS_MAPPINGS
        layerutility_imagescalerestore_v2 = layerstyle_NODE_CLASS_MAPPINGS["ImageScaleRestoreV2"]()
        layerutility_imagescalebyaspectratio_v2 = layerstyle_NODE_CLASS_MAPPINGS["ImageScaleByAspectRatioV2"]()
        layerutility_imagemaskscaleasv2 = layerstyle_NODE_CLASS_MAPPINGS["ImageMaskScaleAsV2"]()


        # 如果没有提供外部模型，使用内部加载的模型
        if model is None or clip is None or vae is None:
            self.initiate()
            model = get_value_at_index(self.checkpointloadersimple_, 0)
            clip = get_value_at_index(self.checkpointloadersimple_, 1)
            vae = get_value_at_index(self.checkpointloadersimple_, 2)

        cliptextencode_14 = cliptextencode.encode(
            text=prompt if prompt else "tryon garment, clothes",
            clip=clip,
        )
        cliptextencode_15 = cliptextencode.encode(
            text="text, watermark,zombie,horror,lowres, embedding:easynegative, nsfw, bad hands",
            clip=clip,
        )
        cliptextencode_27 = cliptextencode.encode(
            text="masterpiece, ((highly detailed)),best quality,\nultra-detailed,\nhires,shadow,\nray tracing, \n\n",
            clip=clip,
        )

# init
        layerutility_imagescalebyaspectratio_v2_43 = (
            layerutility_imagescalebyaspectratio_v2.image_scale_by_aspect_ratio(
                aspect_ratio="original",
                proportional_width=1,
                proportional_height=1,
                fit="fill",
                method="lanczos",
                round_to_multiple="8",
                scale_to_side="longest",
                scale_to_length=1024,
                background_color="#000000",
                image=get_value_at_index(image, 0),
                mask=get_value_at_index(mask, 0),
            )
        )

        conditioningaverage_16 = conditioningaverage.addWeighted(
            conditioning_to_strength=1,
            conditioning_to=get_value_at_index(cliptextencode_27, 0),
            conditioning_from=get_value_at_index(cliptextencode_14, 0),
        )

        controlnetapply_3 = conditioningaverage_16
        if controlnet_image is not None and controlnet is not None:
            layerutility_imagemaskscaleasv2_28 = (
                layerutility_imagemaskscaleasv2.image_mask_scale_as_v2(
                    fit="letterbox",
                    method="lanczos",
                    background_color="#FFFFFF",
                    scale_as=get_value_at_index(
                        layerutility_imagescalebyaspectratio_v2_43, 0
                    ),
                    image=controlnet_image,
                )
            )
            controlnetapply_3 = controlnetapply.apply_controlnet(
                strength=1,
                conditioning=get_value_at_index(conditioningaverage_16, 0),
                control_net=controlnet,
                image=get_value_at_index(layerutility_imagemaskscaleasv2_28, 0),
            )

        ipadapterunifiedloader_5 = ipadapterunifiedloader.load_models(
            preset="LIGHT - SD1.5 only (low strength)",
            model=model,
        )
# ---ref process --
        ipadapteradvanced_11 = ipadapteradvanced.apply_ipadapter(
            weight=1.0,
            weight_type="strong style transfer",
            combine_embeds="concat",
            start_at=0,
            end_at=0.5,
            embeds_scaling="V only",
            model=get_value_at_index(ipadapterunifiedloader_5, 0),
            ipadapter=get_value_at_index(ipadapterunifiedloader_5, 1),
            image=reference,
            attn_mask=reference_mask,
        )

        ipadapterunifiedloader_6 = ipadapterunifiedloader.load_models(
            preset="STANDARD (medium strength)",
            model=get_value_at_index(ipadapteradvanced_11, 0),
        )

        ipadapteradvanced_23 = ipadapteradvanced.apply_ipadapter(
            weight=0.5,
            weight_type="linear",
            combine_embeds="concat",
            start_at=0,
            end_at=0.5,
            embeds_scaling="V only",
            model=get_value_at_index(ipadapterunifiedloader_6, 0),
            ipadapter=get_value_at_index(ipadapterunifiedloader_6, 1),
            image=reference,
        )

        freeu_v2_26 = freeu_v2.patch(
            b1=1.5,
            b2=1.6,
            s1=0.9,
            s2=0.2,
            model=get_value_at_index(ipadapteradvanced_23, 0),
        )

        growmask_25 = growmask.expand_mask(
            expand=grow_mask,
            tapered_corners=True,
            mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_43, 1),
        )

        inpainteasymodel_22 = inpainteasymodel.combine_conditioning(
            strength=1.0,
            start_percent=0,
            end_percent=1,
            positive=get_value_at_index(controlnetapply_3, 0),
            negative=get_value_at_index(cliptextencode_15, 0),
            inpaint_image=get_value_at_index(
                layerutility_imagescalebyaspectratio_v2_43, 0
            ),
            mask=get_value_at_index(growmask_25, 0),
            vae=vae,
        )

        ksampler_19 = ksampler.sample(
            seed=random.randint(1, 2**64) if seed == -1 else seed,
            steps=27,
            cfg=4,
            sampler_name="dpmpp_2m",
            scheduler="karras",
            denoise=denoise,
            model=get_value_at_index(freeu_v2_26, 0),
            positive=get_value_at_index(inpainteasymodel_22, 0),
            negative=get_value_at_index(inpainteasymodel_22, 1),
            latent_image=get_value_at_index(inpainteasymodel_22, 2),
        )

        vaedecode_10 = vaedecode.decode(
            samples=get_value_at_index(ksampler_19, 0),
            vae=vae,
        )

        layerutility_imagescalerestore_v2_50 = (
            layerutility_imagescalerestore_v2.image_scale_restore(
                scale=1,
                method="lanczos",
                scale_by="by_scale",
                scale_by_length=1024,
                image=get_value_at_index(vaedecode_10, 0),
                mask=get_value_at_index(
                    layerutility_imagescalebyaspectratio_v2_43, 1
                ),
                original_size=get_value_at_index(
                    layerutility_imagescalebyaspectratio_v2_43, 2
                ),
            )
        )
        return layerutility_imagescalerestore_v2_50



class TryonV1PoseMask:
    def __init__(self, ):
        self.name = self.__class__.__name__
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        self.checkpointloadersimple_ = checkpointloadersimple.load_checkpoint(
            ckpt_name="juggernaut_reborn.safetensors"
        )
        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        self.controlnetloader_pose = controlnetloader.load_controlnet(
            control_net_name="control_v11p_sd15_openpose.pth"
        )
        clipsetlastlayer = NODE_CLASS_MAPPINGS["CLIPSetLastLayer"]()
        self.clipsetlastlayer_794 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-2,
            clip=get_value_at_index(self.checkpointloadersimple_, 1),
        )
        self.repaintv1 = RepaintV1()

    def forward(self, input_image, input_mask, reference_image, reference_mask, prompt=None):
#     input_image, input_mask
        # input_image =None
        # input_mask=None
        # reference_image = None
        # reference_mask=None
        if not prompt:
            prompt = 'best quality,Ultra-sharp, completely in-focus, no depth of field blur, 8k resolution, hyper-detailed textures, professional studio lighting'

        easy_ismaskempty = NODE_CLASS_MAPPINGS["easy isMaskEmpty"]()
        aio_preprocessor = NODE_CLASS_MAPPINGS["AIO_Preprocessor"]()
        clothessegment = NODE_CLASS_MAPPINGS["ClothesSegment"]()

        aio_preprocessor_810 = aio_preprocessor.execute(
            preprocessor="DWPreprocessor",
            resolution=512,
            image=input_image,
        )
        repaintv1_792 = self.repaintv1.forward(
            seed=random.randint(1, 2**64),
            denoise=1,
            prompt=prompt,
            image=input_image,
            mask=input_mask,
            reference=reference_image,
            reference_mask=reference_mask,
            controlnet_image=get_value_at_index(aio_preprocessor_810, 0),
            controlnet=get_value_at_index(self.controlnetloader_pose, 0),
            model=get_value_at_index(self.checkpointloadersimple_, 0),
            clip=get_value_at_index(self.clipsetlastlayer_794, 0),
            vae=get_value_at_index(self.checkpointloadersimple_, 2),
        )

        # clothessegment_811 = 
        class_selections =  {
            'Hat': False,  'Hair': False,  'Face': False, 'Sunglasses': False,   #  # 帽子  头发  脸 太阳镜 
            'Upper-clothes': True,   'Skirt': True,
            'Dress': True,                              #     连衣裙
            'Belt': True,                               #     腰带
            'Pants': True,  'Left-arm': False, 'Right-arm': False,     #   长裤     左臂   右臂  
            'Left-leg': False, 'Right-leg': False,                     #   左腿     右腿 
            'Bag': False,                                             #    包
            'Scarf': False,                                           #    围巾
            'Left-shoe': False,                                       #    左鞋
            'Right-shoe': False,                                      #    右鞋
            'Background': False,                                     #      background
            'process_res': 512, 'mask_blur': 0, 'mask_offset': 0,
            'background':"Alpha",
            'background_color':"#ffffff",
            'invert_output': False,
            'images': input_image,
        }
        clothessegment_247 = clothessegment.segment_clothes(**class_selections)
        # maskcomposite = NODE_CLASS_MAPPINGS["MaskComposite"]()
        # masks_add_257 = maskcomposite.combine(
        #     x=0,
        #     y=0,
        #     operation="or",
        #     destination=get_value_at_index(clothessegment_247, 1),
        #     source=input_mask,
        # )
        # zdxmaskadd_814 = zdxmaskadd.subtract_masks(
        #     masks_a=get_value_at_index(clothessegment_247, 1),
        #     masks_b=input_mask,
        # )
        return (get_value_at_index(clothessegment_247, 1), )
