import os
from PIL import Image
from .imagefunc import AnyType, log, tensor2pil, pil2tensor

NODE_NAME = 'ColorImage V2'

any = AnyType("*")


def load_custom_size() -> list:
    custom_size_file = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), "custom_size.ini")
    ret_value = ['1024 x 1024',
                '768 x 512',
                '512 x 768',
                '1280 x 720',
                '720 x 1280',
                '1344 x 768',
                '768 x 1344',
                '1536 x 640',
                '640 x 1536'
                 ]
    try:
        with open(custom_size_file, 'r') as f:
            ini = f.readlines()
            for line in ini:
                if not line.startswith(f'#'):
                    ret_value.append(line.strip())
    except Exception as e:
        pass
        # log(f'Warning: {custom_size_file} not found' + f", use default size. ")
    return ret_value

class ColorImageV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        size_list = ['custom']
        size_list.extend(load_custom_size())
        return {
            "required": {
                "size": (size_list,),
                "custom_width": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "custom_height": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "color": ("STRING", {"default": "#000000"},),
            },
            "optional": {
                "size_as": (any, {}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = 'color_image_v2'
    CATEGORY = 'zdx/LayerUtility'

    def color_image_v2(self, size, custom_width, custom_height, color, size_as=None ):

        if size_as is not None:
            if size_as.shape[0] > 0:
                _asimage = tensor2pil(size_as[0])
            else:
                _asimage = tensor2pil(size_as)
            width, height = _asimage.size
        else:
            if size == 'custom':
                width = custom_width
                height = custom_height
            else:
                try:
                    _s = size.split('x')
                    width = int(_s[0].strip())
                    height = int(_s[1].strip())
                except Exception as e:
                    log(f"Warning: {NODE_NAME} invalid size, check {custom_size_file}", message_type='warning')
                    width = custom_width
                    height = custom_height

        ret_image = Image.new('RGB', (width, height), color=color)
        return (pil2tensor(ret_image), )
