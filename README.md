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