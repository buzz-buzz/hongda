```
apt install imagemagick 
apt install python3-pip
sudo apt-get install gcc dialog
sudo apt-get install  swig libpulse-dev # For pip3 install pocketsphinx to success on ubuntu
```
For Windows
===========
```
pip3 install wheel
pip3 install ./third_party/pocketsphinx-0.1.3-cp36-cp36m-win_amd64.whl
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
```
pip3 install -r requirements.txt -i https://pypi.douban.com/simple/
python3 hongda/app.py
```

Run on production
=================
```
pip3 install -r requirements.txt -i https://pypi.douban.com/simple/

python3 hongda/daemon_app.py
```

mount nas
===================
```
sudo apt install nfs-common
mkdir -p /mnt/br-nas
sudo mount -t nfs4 0b42549a88-rmp77.cn-hangzhou.nas.aliyuncs.com:/ /mnt/br-nas
```