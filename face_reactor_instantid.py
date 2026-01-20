# must run at custom_nodes after custom nodes loaded!
import random
import copy
import torch
from PIL import Image, ImageFilter
from typing import Sequence, Mapping, Any, Union
from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODES
import types
from functools import wraps
from .layerstyle import (
    ImageScaleRestore, ImageScaleRestoreV2,
    CropByMask,  CropByMaskV2, 
    RestoreCropBox,
)
# from workflows.src.util import get_value_at_index, tensor2pil, pil2tensor
# from workflows.server_util.service_template import ServiceTemplate
from .annotator import FaceDetector
from .mask import isMaskEmpty, MaskAdd
from .focus_crop_match import FocusCropUltra
import numpy as np

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


easy_ismaskempty = isMaskEmpty()
imagescalerestorev2 = ImageScaleRestoreV2()
restorecropbox = RestoreCropBox()
facedetector = FaceDetector()
focuscropultra = FocusCropUltra()



# models_dir = folder_paths.models_dir
# REACTOR_MODELS_PATH = os.path.join(models_dir, "reactor")
def get_restorers():
    return ["codeformer-v0.1.0.pth", "none", "GFPGANv1.3.pth", "GFPGANv1.4.pth", "GPEN-BFR-512.onnx"]
face_class_section = {
    'Skin':True, 'Nose':True,  'Eyeglasses':True,  'Left-eye':True, 'Right-eye':True, 
    'Left-eyebrow':True,  'Right-eyebrow':True, 'Left-ear':True,  'Right-ear':True, 
    'Mouth':True,  'Upper-lip':True,  'Lower-lip':True, 'Hair':False,  'Earring':True, 
    'Neck':False, 'process_res':512,  'mask_blur':0,  'mask_offset':0, 'invert_output':False, 
    'background':"Alpha", 'images':None,
}

class InstantFaceSwap():
    """
    需要模型:Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors
    controlnet: controlnet-instantid  # ip-adapter.bin" or control_instant_id_sdxl.safetensors
    ipadapter: PLUS FACE (portraits)
    
    """
    def __init__(self):
        self.refiner = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "init_face": ("IMAGE",), "ref_motel_face": ("IMAGE",),
                        "steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}), "is_reactor": ("BOOLEAN", {"default": False}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                        "denoise": ("FLOAT", {"default": 0.8,"min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                    "optional" : {
                        "model": ("MODEL",),
                        "clip": ("CLIP",),
                        "vae": ("VAE",),
                        "prompt": ("STRING", {"multiline": True}),
                        # "swap_model": (list(model_names().keys()),),
                        "face_restore_model": (get_restorers(),),
                        "face_restore_visibility": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1, "step": 0.05}),
                        "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
                    },
                }
    CATEGORY = "zdx/samping"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "forward"
    DESCRIPTION = """
swaper face using sdxl juggernaut and reactor face swap
"""
    def initiate(self):
        self.name = self.__class__.__name__
        checkpointloadersimple = GLOBAL_NODES["CheckpointLoaderSimple"]()
        # diffcontrolnetloader = GLOBAL_NODES["DiffControlNetLoader"]()
        # upscalemodelloader = GLOBAL_NODES["UpscaleModelLoader"]()
        cliptextencode = GLOBAL_NODES["CLIPTextEncode"]()

        self.checkpointloadersimple_xl = checkpointloadersimple.load_checkpoint(
            ckpt_name="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
        )
        # diffcontrolnetloader_286 = diffcontrolnetloader.load_controlnet(
        #     control_net_name="control_instant_id_sdxl.safetensors",
        #     model=get_value_at_index(self.checkpointloadersimple_xl, 0),
        # )
        self.model = get_value_at_index(self.checkpointloadersimple_xl, 0)
        self.clip = get_value_at_index(self.checkpointloadersimple_xl, 1)
        self.vae = get_value_at_index(self.checkpointloadersimple_xl, 2)

        self.cond1 = cliptextencode.encode(
            text="face portrait, best quality,4k", clip=self.clip
        )
        self.cond2 = cliptextencode.encode(
            text="text, watermark, blur, noise",
            clip=self.clip,
        )

    @torch.inference_mode()
    def forward(self, init_face, ref_motel_face, steps=12, is_reactor=False, seed=-1, denoise=0.8, 
            prompt=None, face_restore_model="codeformer-v0.1.0.pth", face_restore_visibility=0.5, codeformer_weight=0.5, model=None, clip=None, vae=None):

        imageupscalewithmodel = GLOBAL_NODES["ImageUpscaleWithModel"]()
        applyinstantidadvanced = GLOBAL_NODES["ApplyInstantIDAdvanced"]()
        inpaintmodelconditioning = GLOBAL_NODES["InpaintModelConditioning"]()
        cliptextencode = GLOBAL_NODES["CLIPTextEncode"]()
        loadimage = GLOBAL_NODES["LoadImage"]()
        growmaskwithblur = GLOBAL_NODES["GrowMaskWithBlur"]()
        ksampler = GLOBAL_NODES["KSampler"]()
        vaedecode = GLOBAL_NODES["VAEDecode"]()
        reactorfaceswap = GLOBAL_NODES["ReActorFaceSwap"]()
        controlnetloader = GLOBAL_NODES["ControlNetLoader"]()
        instantidfaceanalysis = GLOBAL_NODES["InstantIDFaceAnalysis"]()
        instantidmodelloader = GLOBAL_NODES["InstantIDModelLoader"]()
        growmask = GLOBAL_NODES["GrowMask"]()
        maskadd = MaskAdd()
        growmaskwithblur = GLOBAL_NODES["GrowMaskWithBlur"]()
        imagebatch = GLOBAL_NODES["ImageBatch"]()        
        invertmask = GLOBAL_NODES["InvertMask"]()
        imageandmaskpreview = GLOBAL_NODES["ImageAndMaskPreview"]()
        facesegment = GLOBAL_NODES["FaceSegment"]()
        ipadapterunifiedloader = GLOBAL_NODES["IPAdapterUnifiedLoader"]()
        ipadapterfaceid = GLOBAL_NODES["IPAdapterFaceID"]()
        colormatch = GLOBAL_NODES["ColorMatch"]()


        if not hasattr(self, "controlnetloader_instantid"):
            self.controlnetloader_instantid = controlnetloader.load_controlnet(control_net_name="control_instant_id_sdxl.safetensors")
            self.instantidfaceanalysis_92 = instantidfaceanalysis.load_insight_face(provider="CUDA")
            self.instantidmodelloader_ipadapter = instantidmodelloader.load_model(instantid_file="ip-adapter.bin")

        cond1 = cond2 = None
        if model is None:
            if hasattr(self, "model") and self.model is None:
                self.initiate()
            model, clip = self.model, self.clip
            vae = self.vae
            cond1, cond2 = self.cond1,self.cond2
        if cond2 is None:
            cond1 = cliptextencode.encode(text="face portrait, best quality,4k", clip=clip)
            cond2 = cliptextencode.encode(text="text, watermark, blur, noise",clip=clip,)
        if prompt is not None and prompt.strip() != '':
            cond1 = cliptextencode.encode(text=prompt, clip=clip)

        loadimage_ref_model = ref_motel_face
        loadimage_init_face = init_face
        # loadimage_ref_model = loadimage.load_image(image="过来人！千万千万不要去遛猫😭😭_2_搬砖日记_来自小红书网页版.jpg")
        # loadimage_init_face = loadimage.load_image(image="美女-绝美颜，半身.jpeg")

        facedetector_295 = facedetector.call(
            fit="all",
            expand_rate=0.5,
            only_one=True,
            invert=False,
            input_image=loadimage_init_face,
        )
        facedetector_269 = facedetector.call(
            fit="all",
            expand_rate=0.5,
            only_one=True,
            invert=False,
            input_image=loadimage_ref_model,
        )
        easy_ismaskempty_289 = easy_ismaskempty.execute(
            mask=get_value_at_index(facedetector_269, 1)
        )

        easy_ismaskempty_298 = easy_ismaskempty.execute(
            mask=get_value_at_index(facedetector_295, 1)
        )
        if get_value_at_index(easy_ismaskempty_289, 0) or get_value_at_index(easy_ismaskempty_298, 0):
            print('>> input init or motel no face found')
            return (loadimage_ref_model,)

        focuscropultra_297 = focuscropultra.crop_by_mask_v2(
            up_keep=0.2,
            down_keep=0.2,
            right_keep=0.2,
            left_keep=0.2,
            aspect_ratio="original",
            fit="fill",
            method="lanczos",
            round_to_multiple="8",
            scale_to_side="longest",
            scale_to_length=1024,
            background_color="#000000",
            image=loadimage_init_face,
            expand=0, blur_radius=0, # fill_holes=False,
            mask=get_value_at_index(facedetector_295, 1),
        )

        growmask_395 = growmask.expand_mask(
            expand=-20,
            tapered_corners=False,
            mask=get_value_at_index(focuscropultra_297, 4),
        )
        invertmask_393 = invertmask.invert(mask=get_value_at_index(growmask_395, 0))
        imageandmaskpreview_392 = imageandmaskpreview.execute(
            mask_opacity=1,
            mask_color="128, 128, 128",
            pass_through=True,
            image=get_value_at_index(focuscropultra_297, 3),
            mask=get_value_at_index(invertmask_393, 0),
        )
        imagebatch_409 = imagebatch.batch(
            image1=get_value_at_index(focuscropultra_297, 3),
            image2=get_value_at_index(imageandmaskpreview_392, 0),
        )


        focuscropultra_300 = focuscropultra.crop_by_mask_v2(
            up_keep=1,
            down_keep=1,
            right_keep=1,
            left_keep=1,
            aspect_ratio="original",
            fit="fill",
            method="lanczos",
            round_to_multiple="8",
            scale_to_side="longest",
            scale_to_length=1280,
            background_color="#000000",
            expand=0, blur_radius=0,  # fill_holes=False,
            image=loadimage_ref_model,
            mask=get_value_at_index(facedetector_269, 1),
        )

        reactorfaceswap_310 = reactorfaceswap.execute(
            enabled=True,
            swap_model="inswapper_128.onnx",
            facedetection="retinaface_resnet50",
            face_restore_model="none",
            face_restore_visibility=1.0,
            codeformer_weight=codeformer_weight,
            detect_gender_input="no",
            detect_gender_source="no",
            input_faces_index="0",
            source_faces_index="0",
            console_log_level=0,
            input_image=get_value_at_index(focuscropultra_300, 3),
            source_image=get_value_at_index(imagebatch_409, 0),
        )
        face_class_section.update({
            'images':get_value_at_index(focuscropultra_300, 3),
            'background_color': "#222222",
        })
        facesegment_396 = facesegment.segment_face(**face_class_section)
        maskadd_398 = maskadd.subtract_masks(
            masks_a=get_value_at_index(focuscropultra_300, 4),
            masks_b=get_value_at_index(facesegment_396, 1),
        )

        growmaskwithblur_270 = growmaskwithblur.expand_mask(
            expand=8,
            incremental_expandrate=0,
            tapered_corners=True,
            flip_input=False,
            blur_radius=2,
            lerp_alpha=1,
            decay_factor=1,
            fill_holes=True,
            mask=get_value_at_index(maskadd_398, 0),
        )
        final_model = copy.deepcopy(model)
        applyinstantidadvanced_271 = applyinstantidadvanced.apply_instantid(
            ip_weight=1,
            cn_strength=0.6,
            start_at=0,
            end_at=0.9,
            noise=0,
            combine_embeds="average",
            instantid=get_value_at_index(self.instantidmodelloader_ipadapter, 0),
            insightface=get_value_at_index(self.instantidfaceanalysis_92, 0),
            control_net=get_value_at_index(self.controlnetloader_instantid, 0),
            image=get_value_at_index(imagebatch_409, 0),
            model=final_model,
            positive=get_value_at_index(cond1, 0),
            negative=get_value_at_index(cond2, 0),
            image_kps=get_value_at_index(reactorfaceswap_310, 0),
            mask=get_value_at_index(growmaskwithblur_270, 0),
        )

        inpaintmodelconditioning_215 = inpaintmodelconditioning.encode(
            noise_mask=True,
            positive=get_value_at_index(applyinstantidadvanced_271, 1),
            negative=get_value_at_index(applyinstantidadvanced_271, 2),
            vae=vae,
            pixels=get_value_at_index(reactorfaceswap_310, 0),
            mask=get_value_at_index(growmaskwithblur_270, 0),
        )

        ipadapterunifiedloader_383 = ipadapterunifiedloader.load_models(
            preset="PLUS FACE (portraits)",
            model=get_value_at_index(applyinstantidadvanced_271, 0),
        )
        ipadapterfaceid_381 = ipadapterfaceid.apply_ipadapter(
            weight=0.7,
            weight_faceidv2=1,
            weight_type="linear",
            combine_embeds="concat",
            start_at=0,
            end_at=1,
            embeds_scaling="V only",
            model=get_value_at_index(ipadapterunifiedloader_383, 0),
            ipadapter=get_value_at_index(ipadapterunifiedloader_383, 1),
            image=get_value_at_index(imagebatch_409, 0),
            attn_mask=get_value_at_index(growmask_395, 0),
        )

        ksampler_217 = ksampler.sample(
            seed=random.randint(1, 2**64) if seed == -1 else seed,
            steps=steps,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            denoise=denoise,
            model=get_value_at_index(ipadapterfaceid_381, 0),
            positive=get_value_at_index(inpaintmodelconditioning_215, 0),
            negative=get_value_at_index(inpaintmodelconditioning_215, 1),
            latent_image=get_value_at_index(inpaintmodelconditioning_215, 2),
        )

        vaedecode_216 = vaedecode.decode(
            samples=get_value_at_index(ksampler_217, 0),
            vae=vae,
        )
        colormatch_410 = colormatch.colormatch(
            method="mkl",
            strength=1.8,
            multithread=False,
            image_ref=get_value_at_index(reactorfaceswap_310, 0),
            image_target=get_value_at_index(vaedecode_216, 0),
        )
        if not is_reactor:
            imagescalerestorev2_317 = imagescalerestorev2.image_scale_restore(
                scale=1,
                method="lanczos",
                scale_by="by_scale",
                scale_by_length=1024,
                image=get_value_at_index(colormatch_410, 0),
                mask=get_value_at_index(growmaskwithblur_270, 0),
                original_size=get_value_at_index(focuscropultra_300, 5),
            )
            growmaskwithblur_408 = growmaskwithblur.expand_mask(
                expand=24,
                incremental_expandrate=0,
                tapered_corners=True,
                flip_input=False,
                blur_radius=18,
                lerp_alpha=1,
                decay_factor=1,
                fill_holes=False,
                mask=get_value_at_index(imagescalerestorev2_317, 1),
            )
            restorecropbox_326 = restorecropbox.restore_crop_box(
                invert_mask=False,
                background_image=loadimage_ref_model,
                croped_image=get_value_at_index(imagescalerestorev2_317, 0),
                crop_box=get_value_at_index(focuscropultra_300, 2),
                croped_mask=get_value_at_index(growmaskwithblur_408, 0),
            )
            return (restorecropbox_326[0], )
        reactorfaceswap_148 = reactorfaceswap.execute(
            enabled=True,
            swap_model="inswapper_128.onnx",
            facedetection="retinaface_resnet50",
            face_restore_model="codeformer-v0.1.0.pth",
            face_restore_visibility=0.5000000000000001,
            codeformer_weight=0.99,
            detect_gender_input="no",
            detect_gender_source="no",
            input_faces_index="0",
            source_faces_index="0",
            console_log_level=0,
            input_image=get_value_at_index(colormatch_410, 0),
            source_image=get_value_at_index(imagebatch_409, 0),
        )
        imagescalerestorev2_357 = imagescalerestorev2.image_scale_restore(
            scale=1,
            method="lanczos",
            scale_by="by_scale",
            scale_by_length=1024,
            image=get_value_at_index(reactorfaceswap_148, 0),
            mask=get_value_at_index(growmaskwithblur_270, 0),
            original_size=get_value_at_index(focuscropultra_300, 5),
        )
        restorecropbox_359 = restorecropbox.restore_crop_box(
            invert_mask=False,
            background_image=loadimage_ref_model,
            croped_image=get_value_at_index(imagescalerestorev2_357, 0),
            crop_box=get_value_at_index(focuscropultra_300, 2),
        )
        return (restorecropbox_359[0], )

NODE_CLASS_MAPPINGS = {
    "InstantFaceSwap": InstantFaceSwap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantFaceSwap": "InstantFaceSwap"
}



# class FaceSwapReactor(ServiceTemplate):
#     def __init__(self, save_format, use_oss=False):
#         self.name = self.__class__.__name__
#         super(FaceSwapReactor, self).__init__(save_format, use_oss)

#         self.face_swapper = InstantFaceSwap()
#         self.face_swapper.initiate()

#     @torch.inference_mode()
#     def forward(self, request, time_string):
#         """
#             业务处理:
#                 change bacground,
#                 replace_with_reference,
#                 replace_with_prompt,
#                 super_upscale
#         """
#         data_json = request if isinstance(request, dict) else request.get_json()
#         init_image = data_json.get('init_image', None)
#         reference_image = data_json.get('reference_image', None)
#         text_prompt = data_json.get('text_prompt', '')
#         app_type = data_json.get('app_type', '')
#         enhance = data_json.get('enhance', False)

#         loadimage_init_1 = self.loadimage.load_image(image=init_image)
#         load_reference_image_2 = self.loadimage.load_image(image=reference_image)

#         result_image = self.face_swapper.forward(
#             init_face=loadimage_init_1[0],
#             ref_motel_face=load_reference_image_2[0],
#             # text_prompt=text_prompt,
#             is_reactor = enhance,
#         )
#         result_image = tensor2pil(result_image[0])
#         res = {'image': result_image}
#         return res
