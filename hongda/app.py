from bottle import route, run, get, post, request
from api.video import burn
from api.subtitle2 import generate_vtt
from api.beautify import recipe_cartoonize, recipe_paster, recipe_paster_nose, recipe_paster_nose_2

import os.path

@route('/hello')
def hello():
    print('hello')
    return "Hello World!"


@route('/healthcheck')
def healthcheck():
    print('ok')
    return "OK"


@post('/burn_subtitle')
def burn_subtitle():
    burn(srtPath=request.json['srtPath'],
         videoPath=request.json['videoPath'], outputPath=request.json['outputPath'])
    return 'done'


@post('/recognize')
def recognize():
    video_path = request.json['videoPath']
    lock_file = os.path.splitext(video_path)[0] + '.lock'

    if os.path.exists(lock_file):
        return open(lock_file, 'r').read()

    with open(lock_file, 'w') as vtt:
        vtt.write('working...')

    recipe_paster_nose_2(video_path)
    generate_vtt(video_path=video_path, vtt_path=None)

    with open(lock_file, 'w') as vtt:
        vtt.write('done')

    return open(lock_file, 'r').read()

if __name__ == '__main__':
    run(host='0.0.0.0', port=10126, debug=True)
