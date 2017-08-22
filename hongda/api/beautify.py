from skimage.feature import gaussian_filter
from moviepy.editor import VideoFileClip

def blur(image):
    return gaussian_filter(image.astype(float), sigma=2)

clip = VideoFileClip(r"C:\Users\Jeff\AppData\Local\Temp\8318652456269066__90582751-0C2D-41BD-957F-4E111C01609A.mp4")
clip_blurred = clip.fl_image(blur)
clip_blurred.write_videofile("blurred_video.mp4")
