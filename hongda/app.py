from bottle import route, run, get, post, request
import urllib
import json
from api.subtitle import burn


@route('/hello')
def hello():
    return "Hello World!"


@route('/healthcheck')
def healthcheck():
    return "OK"


@post('/burn_subtitle')
def burn_subtitle():
    burn(srtPath=request.json['srtPath'],
         videoPath=request.json['videoPath'], outputPath=request.json['outputPath'])
    return 'done'


run(host='localhost', port=10126, debug=True)
