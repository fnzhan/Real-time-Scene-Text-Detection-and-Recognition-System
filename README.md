# Real-time-Scene-Text-Detection-and-Recognition-System
End-to-end pipeline for real-time scene text detection and recognition.

The detection model use the EAST, the recognition model use the crnn.


# Installation
Please follow the installation guided of [EAST](https://github.com/argman/EAST) and [CRNN](https://github.com/meijieru/crnn.pytorch).

Download trained EAST model from https://drive.google.com/file/d/0B3APw5BZJ67ETHNPaU9xUkVoV0U/view and put it in EAST/result.

Download trained crnn model from https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth?dl=0 and put it in crnn/samples.

# Inference
Run the command:
````bash
cd EAST
bash eval.sh
````

# Screenshot
![image](https://github.com/fnzhan/Real-time-Scene-Text-Detection-and-Recognition-System/blob/master/screenshot.png)
