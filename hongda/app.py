from bottle import route, run, get, post, request

from api.subtitle import burn


@route('/hello')
def hello():
    return "Hello World!"


@route('/healthcheck')
def healthcheck():
    return "OK"


@post('/burn_subtitle')
def burn_subtitle():
    srtPath =request.forms.get('srtPath')
    videoPath =request.forms.get('videoPath')
    outputPath = request.forms.get('outputPath')
    burn(srtPath=srtPath, videoPath=videoPath, outputPath=outputPath)
    return 'done'


run(host='localhost', port=10126, debug=True)
