from scipy.ndimage.filters import gaussian_filter
from moviepy.editor import VideoFileClip
from PIL import Image, ImageEnhance
from numpy import array
from moviepy import *
import moviepy.video.fx.all as vfx


def blur(image):
    return gaussian_filter(image.astype(float), sigma=2)


def contrast(image):
    return array(ImageEnhance.Contrast(Image.fromarray(image)))


def sharp(image):
    return array(ImageEnhance.Sharpness(Image.fromarray(image)).enhance(7.0))


def bright(image):
    brightness = ImageEnhance.Brightness(Image.fromarray(image))
    r = brightness.enhance(2.0)
    return array(r)


clip = VideoFileClip(r"C:\Users\Jeff\AppData\Local\Temp\8318652456269066__90582751-0C2D-41BD-957F-4E111C01609A.mp4",
                     target_resolution=(640, 480))
# clip_blurred = clip.fl_image(blur)
# clip_processed = clip.fl_image(contrast)
# clip_processed = clip.fl_image(contrast)
clip_processed = clip.fl_image(bright)
# clip_processed = clip.fx(vfx.painting, saturation=1, black=0.006)
clip_processed.write_videofile("blurred_video.mp4")
