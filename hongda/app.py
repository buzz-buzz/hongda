from bottle import route, run, get, post, request
from api.subtitle import burn
from api.subtitle2 import generate_vtt
from api.beautify import recipe_cartoonize


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
    generate_vtt(video_path=video_path, vtt_path=None)
    recipe_cartoonize(video_path)
    return 'done'

if __name__ == '__main__':
    run(host='0.0.0.0', port=10126, debug=True)
