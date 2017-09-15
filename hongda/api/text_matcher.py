import difflib
import re


def text_match(expected_vtt, actual_vtt):
    seq = difflib.SequenceMatcher(None, actual_vtt, expected_vtt)
    return seq.ratio()

def get_vtt_content(path):
    with open(path, 'r') as f:
        file_content = f.read()

    m = re.search('WEBVTT[\r\n\s]+\d\d\:\d\d\:\d\d\.\d\d\d\s+\-\-\>\s+\d\d\:\d\d\:\d\d\.\d\d\d[\r\n\s]+([^\r\n]+)[\r\n]', file_content)
    return m.group(1)

def vtt_content_match(expected_vtt_path, actual_vtt_path):
    expected_vtt = get_vtt_content(expected_vtt_path)
    actual_vtt = get_vtt_content(actual_vtt_path)
    return text_match(expected_vtt, actual_vtt)
