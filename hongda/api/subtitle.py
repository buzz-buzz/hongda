from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import *


def generator(txt): return TextClip(txt, font='Georgia-Regular',
                                    fontsize=24, color='white')


def burn(srtPath, videoPath, outputPath):
    sub = SubtitlesClip(srtPath, generator)
    myvideo = VideoFileClip(videoPath)
    final = CompositeVideoClip([myvideo, sub.set_pos(('center', 'bottom'))])
    final.to_videofile(outputPath, fps=myvideo.fps)
