import cv2
from typing import Union
import os
from PIL import Image
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from gfpgan import GFPGANer

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

class SR_Webtoon:
    def __init__(self,
                 model_name_or_path:str="RealESRGAN_x4plus_anime_6B",
                 outscale:int=4,
                 suffix:str='out',
                 tile:int=0,
                 tile_pad:int=10,
                 pre_pad:int=0,
                 face_enhance:bool=False,
                 fp32:bool=False,
                 alpha_upsampler:bool='realesrgan',
                 ext:str='auto',
                 gpu_id:Union[str,None]=None):
        """
        Parameters
        ----------
        model_name_or_path : str
            Model names: RealESRGAN_x4plus_anime_6B
        outscale : float
            The final upsampling scale of the image
        suffix : str
            Suffix of the restored image
        tile : int
            Tile size, 0 for no tile during testing
        tile_pad : int
            Tile padding
        pre_pad : int
            Pre padding size at each border
        face_enhance : bool
            Use GFPGAN to enhance face
        fp32 : bool
            Use fp32 precision during inference. Default: fp16 (half precision).
        alpha_upsampler : str
            The upsampler for the alpha channels. Options: realesrgan | bicubic
        ext : str
            Image extension. Options: auto | jpg | png, auto means using the same extension as inputs
        gpu_id : int
            gpu device to use (default=None) can be 0,1,2 for multi-gpu
        """
        self.__model_name_or_path = model_name_or_path
        self.__outscale = outscale
        self.__suffix = suffix
        self.__tile = tile
        self.__tile_pad = tile_pad
        self.__pre_pad = pre_pad
        self.__face_enhance = face_enhance
        self.__fp32 = fp32
        self.__alpha_upsampler = alpha_upsampler
        self.__ext = ext
        self.__gpu_id = gpu_id

        # model select
        self.__upsampler = self.__model_select()

        # face
        if self.__face_enhance:
            self.__upsampler = self.__face_enhancer()


    def __model_select(self)->RealESRGANer:
        if self.__model_name_or_path == "RealESRGAN_x4plus_anime_6B":
            # download
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']

            # model file format
            model_path = os.path.join('weights', self.__model_name_or_path+ '.pth')
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    # model_path will be updated
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

        elif self.__model_name_or_path is not None:
            model_path = self.__model_name_or_path
        else:
            raise ValueError("model_name_or_path must be RealESRGAN_x4plus_anime_6B")

        # RealESRGAN_x4plus_anime_6B
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4

        # restorer
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=self.__tile,
            tile_pad=self.__tile_pad,
            pre_pad=self.__pre_pad,
            half=not self.__fp32,
            gpu_id=self.__gpu_id)
        return upsampler

    def __face_enhancer(self)->GFPGANer:
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=self.__outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.__upsampler)
        return face_enhancer

    def sr(self,image:Image):
        # image processing
        image = image.convert("RGBA")
        img = np.array(image)

        if self.__face_enhance:
            _, _, output = self.__upsampler.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = self.__upsampler.enhance(img, outscale=self.__outscale)

        sr_img = Image.fromarray(output)
        return sr_img

# test

# if __name__ == "__main__":
#     SR = SR_Webtoon()
#     img = Image.open("/content/004_nocap.jpg").resize((512,512))
#     print(img)
#     r = SR.sr(img)
#     r.save("./test1.png")