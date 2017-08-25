```
apt install imagemagick 
apt install python3-pip
sudo apt-get install gcc dialog
sudo apt-get install swig libpulse-dev # For pip3 install pocketsphinx to success on ubuntu
sudo apt-get install portaudio19-dev
sudo apt-get install liblapack-dev 
sudo apt-get install gfortran
pip3 install opencv-python -i 
```
For Windows
===========
http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-image

```
pip3 install wheel
pip3 install ./third_party/Pillow-4.2.1-cp36-cp36m-win_amd64.whl
pip3 install ./third_party/pocketsphinx-0.1.3-cp36-cp36m-win_amd64.whl
pip3 install ./third_party/numpy-1.13.1-cp36-cp36m-win_amd64.whl
pip3 install ./third_party/scipy-0.19.1-cp36-cp36m-win_amd64.whl
pip3 install ./third_party/scikit_image-0.13.0-cp36-cp36m-win_amd64.whl
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

install ffmpeg on ubuntu
===================
http://johnvansickle.com/ffmpeg/

Download and extract the latest archive:

```
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-64bit-static.tar.xz
tar xvf ffmpeg-git-*.tar.xz
```

Place the ffmpeg and other binaries whereever you need to run that command, for example into /usr/local/bin in order to have it to your $PATH:
```
cd ./ffmpeg-git-*
sudo cp ff* qt-faststart /usr/local/bin/
```
Run ffmpeg using: ffmpeg