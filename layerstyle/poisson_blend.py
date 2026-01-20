import tkinter
from PIL import Image
import numpy as np
from scipy.sparse import csr_matrix
from pyamg.gallery import poisson
from pyamg import ruge_stuben_solver
import matplotlib.pyplot as plt
from skimage.draw import polygon
import torch



def pil2tensor(image):
    new_image = image.convert('RGB')
    img_array = np.array(new_image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)[None]
    return img_tensor

def tensor2pil(image):
    if len(image.shape) < 3:
        image = image.unsqueeze(0)
    return Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))



def getImagePathFromUser(msg):
    tkinter.Tk().withdraw()
    return tkinter.filedialog.askopenfilename(title=msg)


def rgbToGrayMat(imgPth):
    gryImg = Image.open(imgPth).convert('L')
    return np.asarray(gryImg)


def getImageFromUser(msg, srcShp=(0, 0)):
    imgPth = getImagePathFromUser(msg)
    rgb = splitImageToRgb(imgPth)
    if not np.all(np.asarray(srcShp) < np.asarray(rgb[0].shape)):
        return getImageFromUser('Open destination image with resolution bigger than ' +
                                str(tuple(np.asarray(srcShp) + 1)), srcShp)
    return imgPth, rgb


def polyMask(imgPth, numOfPts=100):
    img = rgbToGrayMat(imgPth)
    plt.figure('source image')
    plt.title('Inscribe the region you would like to blend inside a polygon')
    plt.imshow(img, cmap='gray')
    pts = np.asarray(plt.ginput(numOfPts, timeout=-1))
    plt.close('all')
    if len(pts) < 3:
        minRow, minCol = (0, 0)
        maxRow, maxCol = img.shape
        mask = np.ones(img.shape)
    else:
        pts = np.fliplr(pts)
        inPolyRow, inPolyCol = polygon(tuple(pts[:, 0]), tuple(pts[:, 1]), img.shape)
        minRow, minCol = (np.max(np.vstack([np.floor(np.min(pts, axis=0)).astype(int).reshape((1, 2)), (0, 0)]),
                                 axis=0))
        maxRow, maxCol = (np.min(np.vstack([np.ceil(np.max(pts, axis=0)).astype(int).reshape((1, 2)), img.shape]),
                                 axis=0))
        mask = np.zeros(img.shape)
        mask[inPolyRow, inPolyCol] = 1
        mask = mask[minRow: maxRow, minCol: maxCol]
    return mask, minRow, maxRow, minCol, maxCol


def splitImageToRgb(imgPth):
    r, g, b = Image.Image.split(Image.open(imgPth))
    return np.asarray(r), np.asarray(g), np.asarray(b)


def cropImageByLimits(src, minRow, maxRow, minCol, maxCol):
    r, g, b = src
    r = r[minRow: maxRow, minCol: maxCol]
    g = g[minRow: maxRow, minCol: maxCol]
    b = b[minRow: maxRow, minCol: maxCol]
    return r, g, b


def keepSrcInDstBoundaries(corner, gryDstShp, srcShp):
    for idx in range(len(corner)):
        if corner[idx] < 1:
            corner[idx] = 1
        if corner[idx] > gryDstShp[idx] - srcShp[idx] - 1:
            corner[idx] = gryDstShp[idx] - srcShp[idx] - 1
    return corner


def topLeftCornerOfSrcOnDst(dstImgPth, srcShp):
    gryDst = rgbToGrayMat(dstImgPth)
    plt.figure('destination image')
    plt.title('Where would you like to blend it..?')
    plt.imshow(gryDst, cmap='gray')
    center = np.asarray(plt.ginput(2, -1, True)).astype(int)
    plt.close('all')
    if len(center) < 1:
        center = np.asarray([[gryDst.shape[1] // 2, gryDst.shape[0] // 2]]).astype(int)
    elif len(center) > 1:
        center = np.asarray([center[0]])
    corner = [center[0][1] - srcShp[0] // 2, center[0][0] - srcShp[1] // 2]
    return keepSrcInDstBoundaries(corner, gryDst.shape, srcShp)


def cropDstUnderSrc(dstImg, corner, srcShp):
    dstUnderSrc = dstImg[
                  corner[0]:corner[0] + srcShp[0],
                  corner[1]:corner[1] + srcShp[1]]
    return dstUnderSrc


def laplacian(array):
    """计算拉普拉斯算子，返回正确形状的数组"""
    flat_array = array.flatten()
    laplacian_matrix = poisson(array.shape, format='csr')
    result = laplacian_matrix @ flat_array
    return result.reshape(array.shape)


def setBoundaryCondition(b, dstUnderSrc):
    b[1, :] = dstUnderSrc[1, :]
    b[-2, :] = dstUnderSrc[-2, :]
    b[:, 1] = dstUnderSrc[:, 1]
    b[:, -2] = dstUnderSrc[:, -2]
    b = b[1:-1, 1: -1]
    return b


def constructConstVector(mask, mixedGrad, dstUnderSrc, srcLaplacianed, srcShp):
    dstLaplacianed = laplacian(dstUnderSrc)
    b = (1 - mixedGrad) * mask * srcLaplacianed + \
        mixedGrad * mask * dstLaplacianed + \
        (mask - 1) * (-1) * dstLaplacianed
    return setBoundaryCondition(b, dstUnderSrc)


def fixCoeffUnderBoundaryCondition(coeff, shape):
    shapeProd = np.prod(np.asarray(shape))
    arangeSpace = np.arange(shapeProd).reshape(shape)
    arangeSpace[1:-1, 1:-1] = -1
    indexToChange = arangeSpace[arangeSpace > -1]
    for j in indexToChange:
        coeff[j, j] = 1
        if j - 1 > -1:
            coeff[j, j - 1] = 0
        if j + 1 < shapeProd:
            coeff[j, j + 1] = 0
        if j - shape[-1] > - 1:
            coeff[j, j - shape[-1]] = 0
        if j + shape[-1] < shapeProd:
            coeff[j, j + shape[-1]] = 0
    return coeff


def constructCoefficientMat(shape):
    a = poisson(shape, format='lil')
    a = fixCoeffUnderBoundaryCondition(a, shape)
    return a


def buildLinearSystem(mask, srcImg, dstUnderSrc, mixedGrad):
    srcLaplacianed = laplacian(srcImg)
    b = constructConstVector(mask, mixedGrad, dstUnderSrc, srcLaplacianed, srcImg.shape)
    a = constructCoefficientMat(b.shape)
    return a, b


def solveLinearSystem(a, b, bShape):
    multiLevel = ruge_stuben_solver(csr_matrix(a))
    x = np.reshape((multiLevel.solve(b.flatten(), tol=1e-10)), bShape)
    x[x < 0] = 0
    x[x > 255] = 255
    return x


def blend(dst, patch, corner, patchShape, blended):
    mixed = dst.copy()
    mixed[corner[0]:corner[0] + patchShape[0], corner[1]:corner[1] + patchShape[1]] = patch
    blended.append(Image.fromarray(mixed))
    return blended


def poissonAndNaiveBlending(mask, corner, srcRgb, dstRgb, mixedGrad):
    poissonBlended = []
    naiveBlended = []
    for color in range(3):
        src = srcRgb[color]
        dst = dstRgb[color]
        dstUnderSrc = cropDstUnderSrc(dst, corner, src.shape)
        a, b = buildLinearSystem(mask, src, dstUnderSrc, mixedGrad)
        x = solveLinearSystem(a, b, b.shape)
        poissonBlended = blend(dst, x, (corner[0] + 1, corner[1] + 1), b.shape, poissonBlended)
        cropSrc = mask * src + (mask - 1) * (- 1) * dstUnderSrc
        naiveBlended = blend(dst, cropSrc, corner, src.shape, naiveBlended)
    return poissonBlended, naiveBlended


def mergeSaveShow(splitImg, ImgTtl):
    merged = Image.merge('RGB', tuple(splitImg))
    merged.save(ImgTtl + '.png')
    merged.show(ImgTtl)

def split_pil_to_rgb_numpy(pil_img):
    """将PIL图像分解为R, G, B numpy数组"""
    # 确保图像是RGB模式，处理RGBA、灰度等其他模式
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')    
    r, g, b = Image.Image.split(pil_img)
    return np.asarray(r), np.asarray(g), np.asarray(b)


def blend_images(src_image, src_mask, init_image, position=None, blend_mode='Poisson', mixed_grad=0.3, scale_factor=1.0):
    """
    优化的图像融合函数，准确模拟poisson_blend函数的处理流程
    
    :param src_image: 源图像 (PIL Image) 待贴图小目标
    :param src_mask: 蒙版数组，与src_image同尺寸，1表示要融合的区域，0表示不融合
    :param init_image: 目标背景图像 (PIL Image)
    :param position: 融合位置 (x, y)，默认为目标图像中心
    :param blend_mode: 融合方式 'Poisson' 或 'Naive'
    :param mixed_grad: 泊松融合的梯度混合因子 (0-1)
    :param scale_factor: 源图像缩放因子，1.0表示原始大小，0.5表示缩小一半，2.0表示放大一倍
    :return: 融合后的PIL图像
    """
    if not isinstance(src_image, Image.Image) or not isinstance(init_image, Image.Image):
        raise TypeError("src_image和init_image必须是PIL Image对象")

    if isinstance(src_mask, Image.Image):
        src_mask = np.array(src_mask) / 255.0
    if not isinstance(src_mask, np.ndarray):
        raise TypeError("src_mask必须是numpy数组")

    if src_image.size != (src_mask.shape[1], src_mask.shape[0]):
        raise ValueError("src_image和src_mask必须具有相同的尺寸")

    # 步骤0: 计算蒙版的边界框 (模拟getbbox) 限制scale_factor 防止超出目标图像边界
    src_mask_pil = Image.fromarray((src_mask * 255).astype(np.uint8), mode='L')
    src_mask_bbox = src_mask_pil.getbbox()
    if src_mask_bbox is None:
        return init_image
    x,y,x_w,y_h = src_mask_bbox
    w,h = x_w - x, y_h - y
    if scale_factor*w > init_image.width or scale_factor*h > init_image.height:
        scale_factor = min(scale_factor, init_image.width / w, init_image.height / h)
    if scale_factor != 1.0:
        # 计算新的尺寸
        new_width = int(src_image.width * scale_factor)
        new_height = int(src_image.height * scale_factor)
        # 缩放源图像
        src_image = src_image.resize((new_width, new_height), Image.LANCZOS)
        # 缩放蒙版 - 先转换为PIL图像，缩放后再转回numpy
        mask_pil = Image.fromarray((src_mask * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize((new_width, new_height), Image.LANCZOS)
        src_mask = np.array(mask_pil) / 255.0

    # 步骤1: 转换为RGB数组 (模拟splitImageToRgb)
    src_rgb = split_pil_to_rgb_numpy(src_image)
    dst_rgb = split_pil_to_rgb_numpy(init_image)

    # 步骤2: 找到蒙版的边界框 (模拟polyMask的边界计算)
    rows, cols = np.where(src_mask > 0.5)
    if rows.size == 0:
        return init_image

    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # 步骤3: 根据蒙版边界裁剪源图像和蒙版 (模拟cropImageByLimits)
    cropped_mask = src_mask[min_row:max_row + 1, min_col:max_col + 1]
    cropped_src_rgb = cropImageByLimits(src_rgb, min_row, max_row + 1, min_col, max_col + 1)

    # 步骤4: 确定在目标图像上的位置 (模拟topLeftCornerOfSrcOnDst)
    src_shape = cropped_src_rgb[0].shape
    dst_shape = dst_rgb[0].shape

    if dst_shape[0] < src_shape[0] or dst_shape[1] < src_shape[1]:
        raise ValueError("目标图像必须大于裁剪后的源区域")

    if position is None:
        center_y, center_x = dst_shape[0] // 2, dst_shape[1] // 2
    else:
        if position[0] > 1 and position[1] > 1:
            position[0] = position[0] / 100
            position[1] = position[1] / 100
        center_x, center_y = int(position[0] * dst_shape[1]), int(position[1] * dst_shape[0])

    corner = [center_y - src_shape[0] // 2, center_x - src_shape[1] // 2]
    corner = keepSrcInDstBoundaries(corner, dst_shape, src_shape)

    # 步骤5: 执行融合 (模拟poissonAndNaiveBlending)
    if blend_mode == 'Naive':
        blended_channels = []
        for i in range(3):
            src_channel = cropped_src_rgb[i].astype(np.float64)
            dst_channel = dst_rgb[i].astype(np.float64)
            dst_under_src = cropDstUnderSrc(dst_channel, corner, src_shape)
            
            # 朴素融合：使用原始公式
            crop_src = cropped_mask * src_channel + (cropped_mask - 1) * (-1) * dst_under_src
            blended_channel_np = dst_channel.copy()
            blended_channel_np[corner[0]:corner[0] + src_shape[0], 
                             corner[1]:corner[1] + src_shape[1]] = crop_src
            blended_channel_np = np.clip(blended_channel_np, 0, 255)
            blended_channels.append(Image.fromarray(blended_channel_np.astype(np.uint8)))
    
    elif blend_mode == 'Poisson':
        blended_channels = []
        for i in range(3):
            src_channel = cropped_src_rgb[i].astype(np.float64)
            dst_channel = dst_rgb[i].astype(np.float64)
            dst_under_src = cropDstUnderSrc(dst_channel, corner, src_shape)
            
            try:
                a, b = buildLinearSystem(cropped_mask, src_channel, dst_under_src, mixed_grad)
                x = solveLinearSystem(a, b, b.shape)
                
                # 关键修复：使用与原始代码相同的位置偏移
                blended_channel_np = dst_channel.copy()
                patch_corner_y = corner[0] + 1  # 原始代码的偏移
                patch_corner_x = corner[1] + 1  # 原始代码的偏移
                blended_channel_np[patch_corner_y:patch_corner_y + b.shape[0],
                                 patch_corner_x:patch_corner_x + b.shape[1]] = x
                
                blended_channel_np = np.clip(blended_channel_np, 0, 255)
                blended_channels.append(Image.fromarray(blended_channel_np.astype(np.uint8)))
                
            except Exception as e:
                print(f"泊松融合失败，回退到朴素融合: {e}")
                # 回退到朴素融合
                crop_src = cropped_mask * src_channel + (1 - cropped_mask) * dst_under_src
                blended_channel_np = dst_channel.copy()
                blended_channel_np[corner[0]:corner[0] + src_shape[0], 
                                 corner[1]:corner[1] + src_shape[1]] = crop_src
                blended_channel_np = np.clip(blended_channel_np, 0, 255)
                blended_channels.append(Image.fromarray(blended_channel_np.astype(np.uint8)))

    else:
        raise ValueError("blend_mode必须是'Poisson'或'Naive'")

    return Image.merge('RGB', tuple(blended_channels))


class PositionImageBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "init_image": ("IMAGE",),
                "source_image": ("IMAGE",),
                "source_mask": ("MASK",),
                "blend_mode": (["Naive"],),
            },
            "optional": {
                "init_mask": ("MASK",),
                # "center_x": ("FLOAT", {"default": 0.5, "min": 0, "max": 1}),
                # "center_y": ("FLOAT", {"default": 0.5, "min": 0, "max": 1}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "max_or_min_match": (["max", "min"], {"default": "min"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cloned_image",)
    CATEGORY = "zdx/image"
    FUNCTION = "blend_image"
    def blend_image(self, init_image, source_image, source_mask, blend_mode, scale_factor=1.0, max_or_min_match="min", init_mask=None):
        # 转换为 PIL 图像
        src_pil = tensor2pil(source_image)
        init_pil = tensor2pil(init_image)    # dst 底图，
        source_mask_pil = tensor2pil(source_mask).convert('L')
        source_mask_np = np.array(source_mask_pil) / 255
        source_mask_bbox = source_mask_pil.getbbox()
        if source_mask_bbox is None:
            raise ValueError("No valid mask pixels found.")
        x1,y1,x2,y2 = source_mask_bbox
        src_obj_w, src_obj_h = x2 - x1, y2 - y1

        # Calculate the center of the mask if center_x and center_y are not provided
        final_center_x = 0.5
        final_center_y = 0.5
        if init_mask is not None:
            init_mask_pil = tensor2pil(init_mask).convert('L')
            init_mask_np = np.array(init_mask_pil) / 255
            init_mask_bbox = init_mask_pil.getbbox()
            # 使用mask白色区域的中心坐标
            mask_indices = np.argwhere(init_mask_np > 0)
            if len(mask_indices) == 0 or init_mask_bbox is None:
                raise ValueError("No valid  init mask pixels found.")
            x1,y1,x2,y2 = init_mask_bbox
            init_mask_w, init_mask_h = x2 - x1, y2 - y1
            # Calculate bounding box center instead of center of mass
            min_y, min_x = mask_indices.min(axis=0)
            max_y, max_x = mask_indices.max(axis=0)
            final_center_x = (min_x + max_x) // 2
            final_center_y = (min_y + max_y) // 2
            final_center_x = float(final_center_x / init_pil.width)
            final_center_y = float(final_center_y / init_pil.height)
            # 应该使用两者的mask 的bbox的宽高来确定scale_factor
            scale_factor_ = min(init_mask_w / src_obj_w, init_mask_h / src_obj_h) if max_or_min_match == "min" else max(init_mask_w / src_obj_w, init_mask_h / src_obj_h)
            scale_factor = scale_factor_ + scale_factor
            # print(f"Mask bounds: min_y={min_y}, max_y={max_y}, min_x={min_x}, max_x={max_x}")
            # print(f"Geometric center: ({final_center_x}, {final_center_y})")
        # scale_factor = min(scale_factor, init_pil.width / src_pil.width, init_pil.height / src_pil.height)
        blended = blend_images(
            src_image = src_pil,   # 原图形~
            src_mask = source_mask_np,
            init_image = init_pil,
            position=[final_center_x, final_center_y],
            blend_mode = blend_mode,
            mixed_grad = 0.3,
            scale_factor = scale_factor,
        )
        return (pil2tensor(blended),)


def main():
    srcImgPth, srcRgb = getImageFromUser('Open source image')
    mask, *maskLimits = polyMask(srcImgPth)
    srcRgbCropped = cropImageByLimits(srcRgb, *maskLimits)
    dstImgPth, dstRgb = getImageFromUser('Open destination image', srcRgbCropped[0].shape)
    corner = topLeftCornerOfSrcOnDst(dstImgPth, srcRgbCropped[0].shape)
    poissonBlended, naiveBlended = poissonAndNaiveBlending(mask, corner, srcRgbCropped, dstRgb, 0.3)
    mergeSaveShow(naiveBlended, 'Naive Blended')
    mergeSaveShow(poissonBlended, 'Poisson Blended')




if __name__ == '__main__':
    # main()
    """blend_images 的使用示例"""
    dest_img_path = '/data/zdx/gallery/under_sea.webp'
    src_img_path = '/data/zdx/gallery/1_old_airplane003.jpg'
    src_mask_path = '/data/zdx/gallery/1_old_airplane003_mask.jpg'
    
    try:
        res = blend_images(
            Image.open(src_img_path),
            np.array(Image.open(src_mask_path).convert('L')) / 255,
            Image.open(dest_img_path),
            blend_mode='Naive',  # 先尝试朴素融合
            # blend_mode='Poisson'
            position = (0, 0.5),  # 自定义位置 xy
            scale_factor = 2.0,
        )
        res.save('/data/zdx/acdrive/tests/1_blend.jpeg')
        print("图像融合成功！")
    except Exception as e:
        print(f"融合失败: {e}")
