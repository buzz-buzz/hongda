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
python3 hongda/app.py

Run on production
=================
python3 hongda/daemon_app.py