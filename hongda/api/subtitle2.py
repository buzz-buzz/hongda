from subprocess import call
import os.path
import speech_recognition as sr


def extract_audio(video_path):
    audio = os.path.splitext(video_path)[0] + '.wav'
    call(['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio])
    return audio


def recognize(video_path):
    audio_path = extract_audio(video_path)
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        print("Reading audio file from {0}...".format(audio_path))
        audio = r.record(source)

    try:
        print("Listening to {0}...".format(audio_path))
        return r.recognize_sphinx(audio)
    except sr.UnknownValueError:
        return 'Could not understand'


def generate_vtt(video_path, vtt_path):
    if vtt_path is None:
        vtt_path = os.path.splitext(video_path)[0] + '.vtt'
    result = recognize(video_path)
    template = """WEBVTT FILE

1
00:00:00.000 --> 00:00:10.375
{dialog}

""".format(dialog=result)
    with open(vtt_path, 'w') as vtt:
        vtt.write(template)