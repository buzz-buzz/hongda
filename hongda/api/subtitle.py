from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import *
from subprocess import call
import os.path

def generator(txt): return TextClip(txt, font='Georgia-Regular',
                                    fontsize=24, color='white')

def convert_mov_to_mp4(videoPath):
    if videoPath.endswith("MOV") or videoPath.endswith("MOv") or videoPath.endswith("MoV") or videoPath.endswith("Mov") or videoPath.endswith("mOV") or videoPath.endswith("mOv") or videoPath.endswith("moV") or videoPath.endswith("mov"):
        mp4 = os.path.splitext(videoPath)[0]+".mp4"
        call(["ffmpeg", "-i", videoPath, "-vcodec", "copy", "-acodec", "copy", mp4])
        return mp4
    else:
        return videoPath

def convert_mp4_to_mov(videoPath):
     if videoPath.endswith("MP4") or videoPath.endswith("Mp4") or videoPath.endswith("mP4") or videoPath.endswith("mp4"):
         mov = os.path.splitext(videoPath)[0] + '.MOV'
         call(["ffmpeg", "-i", videoPath, "-vcodec", "copy", "-acodec", "copy", mov])
         return mov
     else:
         return videoPath

def burn(srtPath, videoPath, outputPath):
    print(srtPath, videoPath, outputPath)
    videoPath = convert_mov_to_mp4(videoPath)
    print(srtPath, videoPath, outputPath)
    sub = SubtitlesClip(srtPath, generator)
    myvideo = VideoFileClip(videoPath)
    final = CompositeVideoClip([myvideo, sub.set_pos(('center', 'bottom'))])
    tempOutput = os.path.splitext(outputPath)[0] + '.mp4'
    final.to_videofile(tempOutput, fps=myvideo.fps)
    convert_mp4_to_mov(tempOutput)
