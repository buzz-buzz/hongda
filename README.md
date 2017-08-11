```
apt install imagemagick 
apt install python3-pip
```
```
python3
import imageio
imageio.plugins.ffmpeg.download()
```
```
vi /etc/ImageMagick-6/policy.xml
```
Comment out all the policies

Run on local
===========
pip3 install -r requirements.txt -i https://pypi.douban.com/simple/
python3 hongda/app.py

Run on production
=================
python3 hongda/daemon_app.py