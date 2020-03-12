"# DSI-Capstone-Project: Digits Translator App" 

Content
1. Problem Statement
2. Development and Production Models
3. AWS Deployment

Problem Statement
1. To build an App for waypoint translation (digit 0 to 9) from speech to text. Envisaged to enhance the App for waypoint translation in the future
2. Translation to be able to function in environments with and without noise

Data Collection
1. 10 sets of Digit of 0, 1, 2, …, 9 without noise (100 sub-total)
2. 10 sets super-imposed with some simulated noise
3. Another 10 sets super-imposed with more simulated noise (300 in total)

Development Models
1. AdaBoost
2. Random Forest Classifier
3. Convolutional Neural Network

Production Machine and Deep Learning Models
1. Random Forest Classifier
2. Convolutional Neural Network

Flask Local Virtual Environment Deployment Configuration
1. User interface - base.html
2. Flask - service.py, start_flask.bat
3. Pretrained Machine Learning Model - rfc.pkl, ss.pkl, pca.pkl 
4. Pretrained Deep Learning Model - cnn.h5
5. Google Speech Recognition API

AWS Deployment Configuration
1. User interface - base.html
2. Flask - service.py
3. Docker - Dockerfile, Requirements.txt
4. Pretrained Machine Learning Model - rfc.pkl, ss.pkl, pca.pkl 
5. Pretrained Deep Learning Model - cnn.h5
6. Google Speech Recognition API

Current Limitations
1. One (1) audio file per request
2. Recording of up to 15 seconds
3. Waypoint of 10 digits (0 to 9). English words (e.g. decimal or dot) not supported yet.
4. Any codec supported by `soundfile` or `audioread` is supposed to work. Tested wav format.
5. Stereo or Mono channel(s), RFC and CNN trained on mono channel.
6. Around 1 second duration per digit
