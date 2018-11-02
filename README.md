# Virtualglass tryon

## Virtual try-on 
This a small app that combines multiple python tutorial. This app probably doesn't scale very well, but it's simple enough. But it works well even on phone as long as you use firefox.

I used this <a src="https://www.smashingmagazine.com/2016/02/simple-augmented-reality-with-opencv-a-three-js/">tutorial</a> to set up the video stream. One the server side we have a tornado app, that waits for an image to be received and then process the image and return back the location of the glass and a video stream.

For the server side javascript, I used this <a src="https://github.com/matasarei/tryonface">project</a>. It's more or less the same app. The main differences is that in this one there is no server-side app. So estimating head pose with is actually not that great.

Finally, and most importantly, I used <a src="https://github.com/patrikhuber/eos">eos</a> to get an accurate head pose estimator. This was the one that was the most accessible for a python user and It's fast enough to get let yoou process up to 20 images per second.

## Regular install
### Requirements:
Preferred method is to install everything from pip/anaconda

```
pip install -r requirements.txt
```

N.B. : For eos and dlib, you may need to install gcc-7 and g++-7.  You will also need to install cmake.
```
CC=`which gcc-7` CXX=`which g++-7` pip install -r requirements.txt
```

### Additional files:
You should create a directory files, and store inside it all the models used by the app:
```
mkdir files
```

The app also use dlib face detector <a src="https://github.com/davisking/dlib-models">download here ``

You will need additional files to get the head pose estimation with eos. You can get those files on the github of the <a src="https://github.com/patrikhuber/eos/tree/master/share">project</a>:
```
sfm_shape_3448.bin
expression_blendshapes_3448.bin
ibug_to_sfm.txt
sfm_3448_edge_topology.json
ibug_to_sfm.txt
sfm_model_contours.json
```

### Usage
You need to edit the file `index.html`and change the variable ws

To launch the app, just run with python 3
```
python application.py
```

Now, all you have to do is go to firefox <http://localhost:8080>, click on start to initiate the demo. There will be a 5 to 15 seconds latency in the beginning, before the glasses appear.
You can click on synced to get your virtualglass synced with the image (there may be a small delay).

## Installation on aws ubuntu
### Install dependencies
```
sudo apt-get update
sudo apt-get -y upgrade gcc
sudo apt-get install -y g++-7 -y cmake -y libsm6 -y libxrender1 -y libfontconfig1
```

### Install anaconda
```
wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh
bash Anaconda3-5.3.0-Linux-x86_64.sh
```

Activate the root environment
```
source /home/ubuntu/anaconda3/bin/activate root
```

### Additional files
```
git clone https://github.com/patrikhuber/eos

git clone https://github.com/davisking/dlib-models
cd ~/dlib-models
bzip2 -dv ~/dlib-models/shape_predictor_68_face_landmarks.dat.bz2
cd ~
```

### Install the project 
```
git clone https://github.com/operman18/virtualglass-tryon
mkdir ~/virtualglass-tryon/files

ln -s ~/dlib-models/shape_predictor_68_face_landmarks.dat ~/virtualglass-tryon/files
ln -s ~/eos/share/* ~/virtualglass-tryon/files

cd virtualglass-tryon
CC=`which gcc-7` CXX=`which g++-7`  pip install -r requirements.txt
```

### Final step 
You should check if your machine is under a firewall, you will need to open port 8080.

Launch the project with 
```
python application.py
```
