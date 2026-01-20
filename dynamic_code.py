import ast
import sys
import torch
from nodes import NODE_CLASS_MAPPINGS   # ComfyUI 全局节点表

# ---------- 工具函数，原样照搬自 ComfyUI ----------
def get_value_at_index(obj, index):
    """ComfyUI 里很多节点返回 tuple，这个辅助函数用来取第 idx 个元素。"""
    try:
        return obj[index]
    except (IndexError, TypeError):
        return obj
# 使用dict 构建一个context 输入输出类型，这样画布连线非常清爽，将模型，condition,图像，蒙版，等都可以按照dict 包含，
# 下游一条线，可以拿到需要的信息，下游可以通过dict set 设置新的类型的变量；
# 当后面有节点需要处理各种信息需要以前最初的节点来处理的时候，直接用context 来读取，不需从最初的节点来连老远的线
# -----------------~设计理念~----------------------
#      image,mask,reference-->context
#      image,mask,reference-->resize_crop_by_mask--> orig_size,bbox-->context
# 需要事先构造一个kong context~
#----------------------------------------
example_code = '''
import os
import cv2
import sys
from PIL import Image, ImageFilter
from typing import Union, List
from PIL import Image, ImageDraw, ImageFont, ImageChops
import numpy as np
from typing import Sequence, Mapping, Any, Union
import torch
def pil2tensor(image):
    img_array = np.array(image.convert('RGB')).astype(np.float32) / 255.0
    return torch.from_numpy(img_array)[None]
def tensor2pil(image):
    image = image.unsqueeze(0) if len(image.shape) < 3 else image
    return Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))
# bbox = im.getbbox()  # w,h = im.size
# x, y, x_w, y_h = bbox
# width, height = x_w - x, y_h - y
# cropped_im = im.crop(bbox)
'''
# ---------- 真正的节点 ----------

class AnyType(str):
    def __eq__(self, _) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False
ANY_TYPE = AnyType("*")
any_type = AnyType("*")
anyType = AnyType("*")
anytype = AnyType("*")


_all_contextput_output_data = {
    "context": ("context", "RUN_CONTEXT", "context"),
    "model": ("model", "MODEL", "model"),
    "positive": ("positive", "CONDITIONING", "positive"),
    "negative": ("negative", "CONDITIONING", "negative"),
    "latent": ("latent", "LATENT", "latent"),
    "vae": ("vae", "VAE", "vae"),
    "clip": ("clip","CLIP", "clip"),
    "images": ("images", "IMAGE", "images"),
    "mask": ("mask", "MASK", "mask"),
    "guidance": ("guidance", "FLOAT", "guidance"),
    "pos": ("pos", "STRING", "pos"),
    "neg": ("neg", "STRING", "neg"),
    "width": ("width", "INT","width" ),
    "height": ("height", "INT","height"),
    "data": ("data", ANY_TYPE, "data"),
    "data1": ("data1", ANY_TYPE, "data1"),
    "data2": ("data2", ANY_TYPE, "data2"),
    "data3": ("data3", ANY_TYPE, "data3"),
    "data4": ("data4", ANY_TYPE, "data4"),
    "data5": ("data5", ANY_TYPE, "data5"),
    "data6": ("data6", ANY_TYPE, "data6"),
    "data7": ("data7", ANY_TYPE, "data7"),
    "data8": ("data8", ANY_TYPE, "data8"),
}
TYPE_LIST = list(_all_contextput_output_data.keys())


def new_context(context, **kwargs):
    context = context if context is not None else None
    new_ctx = {}
    for key in _all_contextput_output_data:
        if key == "context":
            continue
        v = kwargs[key] if key in kwargs else None
        new_ctx[key] = v if v is not None else (
            context[key] if context is not None and key in context else None
        )
    return new_ctx


class ContextData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    },
            "optional": {
                "context": ("RUN_CONTEXT",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "images": ("IMAGE",),
                "mask": ("MASK",), 
                "orig_size": ("original_size",),
                "crop_box": ("BOX",),
                "data1": (ANY_TYPE,),
                "data2": (ANY_TYPE,),
                "data3": (ANY_TYPE,),
                "data4": (ANY_TYPE,),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "CONDITIONING","CONDITIONING","LATENT","VAE","CLIP","IMAGE","MASK",)
    RETURN_NAMES = ("context", "model","positive","negative","latent","vae","clip","images","mask",)
    FUNCTION = "sample"
    CATEGORY = "zdx/context"
    def sample(self, context=None,model=None,positive=None,negative=None,latent=None,vae =None,clip =None,
            images =None, mask =None, data1=None, data2=None, data3=None, data4=None):
        if not context:
            context = new_context(context,model=model,positive=positive,negative=negative,latent=latent,vae=vae,clip=clip,images=images,mask=mask,)
        if model is None:
            model = context.get("model")
        if positive is None:
            positive = context.get("positive")
        if negative is None:
            negative = context.get("negative")
        if vae is None:
            vae = context.get("vae")
        if clip is None:
            clip = context.get("clip")
        if latent is None:
            latent = context.get("latent")
        if images is None:
            images = context.get("images")
        if mask is None:
            mask = context.get("mask")
        context = new_context(context,model=model,positive=positive,negative=negative,latent=latent,vae=vae,clip=clip,images=images,mask=mask,
                data1=data1, data2=data2, data3=data3, data4=data4,)
        return (context, model, positive, negative, latent, vae, clip, images, mask,)

class AnythingToContext:
    # 把任意数据按照kv的形式注入context，便于在下游节点使用它
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "key": ("STRING", {"default": "data"}), "value": (ANY_TYPE,),
                "key1": ("STRING", {"default": "data_key1"}), "value1": (ANY_TYPE,),
            },
        }
    RETURN_TYPES = ("RUN_CONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "set"
    CATEGORY = "zdx/context"
    def set(self, context=None, key="data", value=None):
        context[key] = value
        return (context,)

# 2. 专用注入节点
class ContextSetImageMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "image1_key": ("STRING", {"default": "image_"}),  "image1": ("IMAGE",),  # 改为下拉选择类型
                "mask1_key": ("STRING", {"default": "mask_"}), "mask1": ("MASK",),
                "crop_box_key": ("STRING", {"default": "crop_box_"}), "crop_box": ("BOX",),
                "image2_key": ("STRING", {"default": "image_"}),  "image2": ("IMAGE",),  # 改为下拉选择类型
                "mask2_key": ("STRING", {"default": "mask_"}), "mask2": ("MASK",),
                "orig_size_key": ("STRING", {"default": "orig_size_"}), "orig_size": ("original_size",),
            }
        }
    RETURN_TYPES = ("RUN_CONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "set"
    CATEGORY = "zdx/context"
    def set(self, context, image1_key=None, image1=None, mask1_key=None, mask1=None,
            crop_box_key=None, crop_box=None, image2_key=None, image2=None, mask2_key=None, mask2=None,
            orig_size_key=None, orig_size=None):
        if image1 is not None and image1_key:
            context[image1_key] = image1
        if mask1 is not None and mask1_key:
            context[mask1_key] = mask1
        if crop_box is not None and crop_box_key:
            context[crop_box_key] = crop_box
        if image2 is not None and image2_key:
            context[image2_key] = image2
        if mask2 is not None and mask2_key:
            context[mask2_key] = mask2
        if orig_size is not None and orig_size_key:
            context[orig_size_key] = orig_size
        return (context,)


class ContextGet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "key": ("STRING", {"default": "data"}),
            },
        }
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
    FUNCTION = "get"
    CATEGORY = "zdx/context"
    def get(self, context=None, key="data"):
        out = context[key]
        return (out,)


class ContextSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "type": (TYPE_LIST, {}),
            },
        }
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
    FUNCTION = "get"
    CATEGORY = "zdx/context"
    def get(self, type, context=None):
        out = context[type]
        return (out,)

class DynamicCode:
    """
    在 UI 里写 Python 代码，直接调用任意已加载节点。和上下文context,并支持向context写入数据。已支持下游使用context。
    代码里可直接使用：
        NODE_CLASS_MAPPINGS / get_value_at_index / torch / PIL / np / …
    以及本节点输入参数：context 字典, image, mask, 
    最后一个表达式的值（或显式 return）会被当作本节点输出。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "code": ("STRING", {"multiline": True, "default": f"# write your code here\n{example_code}\noutput = (None, image)"}),
            },
            "optional": {
                "context": ("RUN_CONTEXT",),   
                "ENABLE": ("BOOLEAN", {"default": True}),
                "image": ("IMAGE",),
                "mask":  ("MASK",),
                "data": (ANY_TYPE,), "data1": (ANY_TYPE,), "data2": (ANY_TYPE,), "data3": (ANY_TYPE,),
                "data4": (ANY_TYPE,)
            },
        }
    _MAX_OUT = 5
    RETURN_TYPES = ("IMAGE","MASK","RUN_CONTEXT")   # 可根据需要扩充
    RETURN_NAMES = ("image1","mask1","context")
    FUNCTION = "execute_code"
    CATEGORY = "zdx/context"
    def execute_code(self,
                     code, context=None, ENABLE=True,
                     image=None, mask=None,  data=None, data1=None, data2=None, data3=None, data4=None):
        # 1. 初始化context，如果为None则创建空字典
        if context is None:
            context = {}
        
        # 2. 准备注入命名空间
        safe_dict = {
            # ComfyUI 节点表
            "NODE_CLASS_MAPPINGS": NODE_CLASS_MAPPINGS,
            # 常用工具
            "get_value_at_index": get_value_at_index,
            "torch": torch,
            # 把当前所有输入参数也放进去，用户代码可直接调用
            "ENABLE": ENABLE, 
            "context": context,  # 传入可修改的context字典
            "image": image, "mask": mask,
            "data": data, 
            "data1": data1, 
            "data2": data2, 
            "data3": data3, 
            "data4": data4,
            "random": __import__("random"), 
            "np": __import__("numpy"), 
            "PIL": __import__("PIL.Image", fromlist=["Image"]),
            # 添加context操作的便捷函数
            "ctx_get": lambda key, default=None: context.get(key, default),
            "ctx_set": lambda key, value: context.update({key: value}),
            "ctx_has": lambda key: key in context,
            "ctx_keys": lambda: list(context.keys()),
            "ctx_clear": lambda: context.clear(),
        }
        # 2. 如果用户代码里有显式 return，我们把它换成局部变量 _return
        try:
            parsed = ast.parse(code)
            last_stmt = parsed.body[-1] if parsed.body else None
            if isinstance(last_stmt, ast.Expr):
                # 把最后一个表达式变成 `_return = <expr>`
                assign = ast.Assign(
                    targets=[ast.Name(id='_return', ctx=ast.Store())],
                    value=last_stmt.value)
                parsed.body[-1] = assign
                compiled_code = compile(parsed, '<dynamic_code>', 'exec')
                need_return = True
            else:
                compiled_code = compile(code, '<dynamic_code>', 'exec')
                need_return = False
        except SyntaxError as e:
            print(f"❌ DynamicCode 语法错误:")
            print(f"   错误位置: 第 {e.lineno} 行, 第 {e.offset} 列")
            print(f"   错误信息: {e.msg}")
            print(f"   问题代码: {e.text.strip() if e.text else '未知'}")
            return (image, mask, context) if image is not None else (None, None, context)
        except Exception as e:
            print(f"❌ DynamicCode 代码解析错误: {e}")
            return (image, mask, context) if image is not None else (None, None, context)
        # 3. 执行代码
        try:
            exec(compiled_code, safe_dict)
        except Exception as e:
            print(f"❌ DynamicCode 执行错误:")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            # 打印调用栈信息
            import traceback
            print(f"   详细错误信息:")
            traceback.print_exc()
            return (image, mask, context) if image is not None else (None, None, context)        
        # 4. 取结果
        try:
            if need_return:
                result = safe_dict.get('_return', image)
            else:
                # 用户写了显式 return
                result = safe_dict.get('result', image)
        except Exception as e:
            print(f"❌ DynamicCode 结果获取错误: {e}")
            result = image
        # 5. 统一成 tuple 格式并返回
        try:
            if not isinstance(result, tuple):
                result = (result,)
            # 确保返回格式正确: (image, mask, context)
            final_result = []
            # 处理第一个返回值 (image)
            if len(result) > 0:
                final_result.append(result[0])
            else:
                final_result.append(image)
            # 处理第二个返回值 (mask)  
            if len(result) > 1:
                final_result.append(result[1])
            else:
                final_result.append(mask)
            # 处理第三个返回值 (context)
            final_result.append(context)
            print(f"✅ DynamicCode 执行成功")
            return tuple(final_result)
        except Exception as e:
            print(f"❌ DynamicCode 结果处理错误: {e}")
            return (image, mask, context) if image is not None else (None, None, context)

class DynamicCodeMask(DynamicCode):
    _MAX_OUT = 5
    RETURN_TYPES = ("MASK","RUN_CONTEXT")   # 可根据需要扩充
    RETURN_NAMES = ("mask1","context")
    FUNCTION = "execute_code"
    CATEGORY = "zdx/context"

# ---------- 注册 ----------
_NODE_CLASS_MAPPINGS = {
    "DynamicCode": DynamicCode,
    "ContextData": ContextData,
    "ContextSelect": ContextSelect,
    "ContextGet": ContextGet,
    "AnythingToContext": AnythingToContext,
    "ContextSetImageMask": ContextSetImageMask,
    # "DynamicCodePlus": DynamicCodePlus,
}


_NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicCode": "zdxDynamicCode",
    # "DynamicCodePlus": "Dynamic Code Plus (抽象节点)",
    "ContextData": "ContextData",
    "ContextSelect": "ContextSelect",
    "ContextGet": "ContextGet",
    "AnythingToContext": "AnythingToContext",
    "ContextSetImageMask": "ContextSetImageMask",
}

