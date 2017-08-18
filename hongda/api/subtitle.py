from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import *
from subprocess import call
import os.path
import subprocess
import shlex
import json


def get_rotation(file_path):
    """
    Function to get the rotation of the input video file.
    Adapted from gist.github.com/oldo/dc7ee7f28851922cca09/revisions using the ffprobe comamand by Lord Neckbeard from
    stackoverflow.com/questions/5287603/how-to-extract-orientation-information-from-videos?noredirect=1&lq=1

    Returns a rotation None, 90, 180 or 270
    """
    cmd = "ffprobe -loglevel error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1"
    args = shlex.split(cmd)
    args.append(file_path)
    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobe_output = subprocess.check_output(args).decode('utf-8')
    if len(ffprobe_output) > 0:  # Output of cmdis None if it should be 0
        ffprobe_output = json.loads(ffprobe_output)
        rotation = ffprobe_output
    else:
        rotation = 0

    return rotation


def get_dimension(file_path):
    cmd = "ffprobe -v quiet -print_format json -show_streams"
    args = shlex.split(cmd)
    args.append(file_path)
    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)

    # find height and width
    height = ffprobeOutput['streams'][0]['height']
    width = ffprobeOutput['streams'][0]['width']

    return height, width


def generator(txt): return TextClip(txt, font='Georgia-Regular',
                                    fontsize=30, color='white')


def convert_mov_to_mp4(videoPath):
    file, ext = os.path.splitext(videoPath)
    if ext.lower() == ".mov":
        mp4 = file + ".mp4"
        if(not os.path.isfile(mp4)):
            call(["ffmpeg", "-i", videoPath, "-vcodec",
                  "copy", "-acodec", "copy", mp4])
            return mp4

    return videoPath


def convert_mp4_to_mov(videoPath):
    file, ext = os.path.splitext(videoPath)

    if ext.lower() == '.mp4':
        mov = file + '.MOV'
        if not os.path.isfile(mov):
            call(["ffmpeg", "-i", videoPath, "-vcodec",
                  "copy", "-acodec", "copy", mov])
            return mov

    return videoPath


def compress_dimension(path):
    h, w = get_dimension(path)
    print("original (h, w): " + str((h, w)))
    r = h / w
    if max(h, w) > 640:
        if h < w:
            w = 640
            h = int(w * r)
        else:
            h = 640
            w = int(h / r)

    print("compressed (h, w): " + str((h, w)))
    return h, w


def compress_dimension_with_rotation_handled(path):
    h, w = compress_dimension(path)
    rotation = get_rotation(path)
    _, ext = os.path.splitext(path)
    if rotation == 90 or rotation == 270:  # If video is in portrait
        print("video is portrait")
        return w, h
    return h, w


def burn(srtPath, videoPath, outputPath):
    d = compress_dimension_with_rotation_handled(videoPath)
    videoPath = convert_mov_to_mp4(videoPath)
    sub = SubtitlesClip(srtPath, generator)
    myvideo = VideoFileClip(
        videoPath, target_resolution=d)

    final = CompositeVideoClip([myvideo, sub.set_pos(('center', 'bottom'))])
    temp_output = os.path.splitext(outputPath)[0] + '.mp4'
    if not os.path.isfile(temp_output):
        final.write_videofile(temp_output, fps=myvideo.fps)
    convert_mp4_to_mov(temp_output)
