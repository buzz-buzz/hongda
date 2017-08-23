from bottle import route, run, get, post, request
from api.subtitle import burn
from api.subtitle2 import generate_vtt
from bottledaemon import daemon_run
import app



if __name__ == '__main__':
    daemon_run(host='0.0.0.0', port=10126)
