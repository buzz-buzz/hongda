from bottle import route, run, get, post, request
import urllib
import json
from api.subtitle import burn
from bottledaemon import daemon_run


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


if __name__ == '__main__':
    daemon_run(host='0.0.0.0', port=10126)
