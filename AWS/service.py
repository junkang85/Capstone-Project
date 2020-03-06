import librosa
import numpy as np
import pandas as pd

from flask import Flask, jsonify, request
from sklearn.externals import joblib

app = Flask(__name__)

cnn = joblib.load('cnn.pkl')

def Resample_1kHz(x):
    return np.array(librosa.resample(x, orig_sr = 44100, target_sr = 1000))

def Cut_1sec(x):
    peak = librosa.util.peak_pick(x, 1000, 1000, 1000, 1000, 0, 1000)
    cut = x[(peak[0] - 500):(peak[0] + 500)]
    pad = librosa.util.pad_center(cut, 1000)
    return pad

def CNN_Single(x):
    digits = pd.DataFrame(x)
    digits.drop(columns = 1, inplace = True)
    digits.rename(columns = {0: 'Speech_Raw'}, inplace = True)
    digits['Speech_1kHz'] = digits['Speech_Raw'].apply(Resample_1kHz)
    digits['Speech_1sec'] = digits['Speech_1kHz'].apply(Cut_1sec)
    X =[]
    for i in range(0,digits.shape[0]):
        X.append(digits.Speech_1sec[i].reshape(1000,1))
    X = np.array(X).reshape(-1,1000,1)
    return cnn.predict_classes(X)[0]

@app.route('/')
def hello_world():
    return 'Welcome to JK Digit Translator App'

@app.route('/predict-digits-interface', methods = ["POST"])
def predict_digits_interface():
    output = None
    if request.method == "POST":
        audio = request.form['audioFile']
        output = CNN_Single(audio)
    return render_template("base.html", output = output)

if __name__ == '__main__':
    app.run(host = "0.0.0.0")