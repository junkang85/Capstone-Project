import librosa
import pandas as pd
import numpy as np
import os
import scipy
import speech_recognition as sr

from flask import Flask, jsonify, request, render_template
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.python.keras.utils import np_utils
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

UPLOAD_FOLDER = 'tmp/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ss = joblib.load('ss.pkl')
pca = joblib.load('pca.pkl')
rfc = joblib.load('rfc.pkl')
cnn = load_model('cnn.h5')

def Resample_1kHz(x):
    return np.array(librosa.resample(x, orig_sr = 44100, target_sr = 1000))
def Cut_1sec(x):
    peak = librosa.util.peak_pick(x, 1000, 1000, 1000, 1000, 0, 1000)
    cut = x[(peak[0] - 500):(peak[0] + 500)]
    pad = librosa.util.pad_center(cut, 1000)
    return pad
def remaining_cut(x):
    peak = librosa.util.peak_pick(x, 1000, 1000, 1000, 1000, 0.1, 1000)
    if len(peak) == 0:
        return 'Nothing to cut'
    if peak <= 500:
        cut = x[0:1000]
        pad = librosa.util.pad_center(cut, 1000)
    if peak > 500:
        cut = x[(peak[0] - 500):(peak[0] + 500)]
        pad = librosa.util.pad_center(cut, 1000)
    return pad
def AbsMean(x):
    out = np.abs(x)
    return out.mean()
def SD(x):
    return x.std()
def Skewness(x):
    return skew(x)
def Kurtosis(x):
    return kurtosis(x, fisher = True)
def RMS(x):
    data = librosa.feature.rms(x, frame_length = 1000, hop_length = 1001)
    return data[0][0]
def Flat(x):
    data = librosa.feature.spectral_flatness(x, hop_length = 1001)
    return data[0][0]
def ZCR(x):
    data = librosa.feature.zero_crossing_rate(x, frame_length = 1000, hop_length = 1001)
    return data[0][0]
def Centroid(x):
    data = librosa.feature.spectral_centroid(x, sr = 1000, hop_length = 1001)
    return data[0][0]
def MFCC1(x):
    data = librosa.feature.mfcc(x, sr = 1000, n_mfcc = 13)
    return data[0].mean()
def MFCC2(x):
    data = librosa.feature.mfcc(x, sr = 1000, n_mfcc = 13)
    return data[1].mean()
def MFCC3(x):
    data = librosa.feature.mfcc(x, sr = 1000, n_mfcc = 13)
    return data[2].mean()
def MFCC4(x):
    data = librosa.feature.mfcc(x, sr = 1000, n_mfcc = 13)
    return data[3].mean()
def MFCC5(x):
    data = librosa.feature.mfcc(x, sr = 1000, n_mfcc = 13)
    return data[4].mean()
def MFCC6(x):
    data = librosa.feature.mfcc(x, sr = 1000, n_mfcc = 13)
    return data[5].mean()
def MFCC7(x):
    data = librosa.feature.mfcc(x, sr = 1000, n_mfcc = 13)
    return data[6].mean()
def MFCC8(x):
    data = librosa.feature.mfcc(x, sr = 1000, n_mfcc = 13)
    return data[7].mean()
def MFCC9(x):
    data = librosa.feature.mfcc(x, sr = 1000, n_mfcc = 13)
    return data[8].mean()
def MFCC10(x):
    data = librosa.feature.mfcc(x, sr = 1000, n_mfcc = 13)
    return data[9].mean()
def MFCC11(x):
    data = librosa.feature.mfcc(x, sr = 1000, n_mfcc = 13)
    return data[10].mean()
def MFCC12(x):
    data = librosa.feature.mfcc(x, sr = 1000, n_mfcc = 13)
    return data[11].mean()
def MFCC13(x):
    data = librosa.feature.mfcc(x, sr = 1000, n_mfcc = 13)
    return data[12].mean()
def LPC1(x):
    LPC = librosa.core.lpc(x, 12)
    return LPC[1]
def LPC2(x):
    LPC = librosa.core.lpc(x, 12)
    return LPC[2]
def LPC3(x):
    LPC = librosa.core.lpc(x, 12)
    return LPC[3]
def LPC4(x):
    LPC = librosa.core.lpc(x, 12)
    return LPC[4]
def LPC5(x):
    LPC = librosa.core.lpc(x, 12)
    return LPC[5]
def LPC6(x):
    LPC = librosa.core.lpc(x, 12)
    return LPC[6]
def LPC7(x):
    LPC = librosa.core.lpc(x, 12)
    return LPC[7]
def LPC8(x):
    LPC = librosa.core.lpc(x, 12)
    return LPC[8]
def LPC9(x):
    LPC = librosa.core.lpc(x, 12)
    return LPC[9]
def LPC10(x):
    LPC = librosa.core.lpc(x, 12)
    return LPC[10]
def LPC11(x):
    LPC = librosa.core.lpc(x, 12)
    return LPC[11]
def LPC12(x):
    LPC = librosa.core.lpc(x, 12)
    return LPC[12]
def RFC_Single():
    pred = None
    digits = [librosa.load('./tmp/single/single.wav', sr = 44100)]
    digits = pd.DataFrame(digits)
    digits.drop(columns = 1, inplace = True)
    digits.rename(columns = {0: 'Speech_Raw'}, inplace = True)
    digits['Speech_1kHz'] = digits['Speech_Raw'].apply(Resample_1kHz)
    digits['Speech_1sec'] = digits['Speech_1kHz'].apply(Cut_1sec)    
    digits['AbsMean'] = digits['Speech_1sec'].apply(AbsMean)
    digits['SD'] = digits['Speech_1sec'].apply(SD)
    digits['Skewness'] = digits['Speech_1sec'].apply(Skewness)
    digits['Kurtosis'] = digits['Speech_1sec'].apply(Kurtosis)
    digits['RMS'] = digits['Speech_1sec'].apply(RMS)
    digits['Flat'] = digits['Speech_1sec'].apply(Flat)
    digits['ZCR'] = digits['Speech_1sec'].apply(ZCR)
    digits['Centroid'] = digits['Speech_1sec'].apply(Centroid)
    digits['MFCC1'] = digits['Speech_1sec'].apply(MFCC1)
    digits['MFCC2'] = digits['Speech_1sec'].apply(MFCC2)
    digits['MFCC3'] = digits['Speech_1sec'].apply(MFCC3)
    digits['MFCC4'] = digits['Speech_1sec'].apply(MFCC4)
    digits['MFCC5'] = digits['Speech_1sec'].apply(MFCC5)
    digits['MFCC6'] = digits['Speech_1sec'].apply(MFCC6)
    digits['MFCC7'] = digits['Speech_1sec'].apply(MFCC7)
    digits['MFCC8'] = digits['Speech_1sec'].apply(MFCC8)
    digits['MFCC9'] = digits['Speech_1sec'].apply(MFCC9)
    digits['MFCC10'] = digits['Speech_1sec'].apply(MFCC10)
    digits['MFCC11'] = digits['Speech_1sec'].apply(MFCC11)
    digits['MFCC12'] = digits['Speech_1sec'].apply(MFCC12)
    digits['MFCC13'] = digits['Speech_1sec'].apply(MFCC13)
    digits['LPC1'] = digits['Speech_1sec'].apply(LPC1)
    digits['LPC2'] = digits['Speech_1sec'].apply(LPC2)
    digits['LPC3'] = digits['Speech_1sec'].apply(LPC3)
    digits['LPC4'] = digits['Speech_1sec'].apply(LPC4)
    digits['LPC5'] = digits['Speech_1sec'].apply(LPC5)
    digits['LPC6'] = digits['Speech_1sec'].apply(LPC6)
    digits['LPC7'] = digits['Speech_1sec'].apply(LPC7)
    digits['LPC8'] = digits['Speech_1sec'].apply(LPC8)
    digits['LPC9'] = digits['Speech_1sec'].apply(LPC9)
    digits['LPC10'] = digits['Speech_1sec'].apply(LPC10)
    digits['LPC11'] = digits['Speech_1sec'].apply(LPC11)
    digits['LPC12'] = digits['Speech_1sec'].apply(LPC12)
    features = ['AbsMean', 'SD', 'Skewness', 'Kurtosis', 'RMS', 'Flat', 'ZCR', 'Centroid', 'LPC1', 'LPC2', 'LPC3', 'LPC4', 'LPC5', 'LPC6', 'LPC7', 'LPC8', 'LPC9', 'LPC10', 'LPC11', 'LPC12', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13']
    X_sc = ss.transform(digits[features])
    Z = pca.transform(X_sc)
    pred = rfc.predict(Z)
    return str(pred[0])

def CNN_Single():
    pred = None
    digits = [librosa.load('./tmp/single/single.wav', sr = 44100)]
    digits = pd.DataFrame(digits)
    digits.drop(columns = 1, inplace = True)
    digits.rename(columns = {0: 'Speech_Raw'}, inplace = True)
    digits['Speech_1kHz'] = digits['Speech_Raw'].apply(Resample_1kHz)
    digits['Speech_1sec'] = digits['Speech_1kHz'].apply(Cut_1sec)
    X =[]
    for i in range(0,digits.shape[0]):
        X.append(digits.Speech_1sec[i].reshape(1000, 1))
    X = np.array(X).reshape(-1, 1000, 1)
    pred = cnn.predict_classes(X)
    return str(pred[0])
def Google_API(x):
    output = None
    r = sr.Recognizer()
    with sr.AudioFile(x) as source:
        google = r.record(source)
    try:
        output = r.recognize_google(google)
    except sr.RequestError:
        output = 'API unavailable or unresponsive'
    except sr.UnknownValueError:
        output = 'Sorry, speech was unintelligible'
    return output

@app.route('/')
def hello_world():
    return 'Welcome to JK Digits Translator App'

@app.route('/digits', methods = ['GET', 'POST'])
def predict_digits():
    audio_Filename = None
    Google_Predict = None
    RFC_Digits = None
    CNN_Digits = None
    if request.method == 'POST':
        audio = request.files['audioFile']
        audio_Filename = audio.filename
        audio.save(app.config['UPLOAD_FOLDER'] + '/' + audio_Filename)
        Google_Predict = Google_API(app.config['UPLOAD_FOLDER'] + '/' + audio_Filename)
        digits = [librosa.load(app.config['UPLOAD_FOLDER'] + '/' + audio_Filename, sr = 44100, mono = True, duration = 15)]
        digits = pd.DataFrame(digits)
        digits.drop(columns = 1, inplace = True)
        digits.rename(columns = {0: 'Speech_Raw'}, inplace = True)
        digits['Speech_1kHz'] = digits['Speech_Raw'].apply(Resample_1kHz)
        counter = 10
        peak = librosa.util.peak_pick(digits['Speech_1kHz'][0], 1000, 1000, 1000, 1000, 0.1, 1000)
        cut = digits['Speech_1kHz'][0][(peak[0] - 500):(peak[0] + 500)]
        pad = librosa.util.pad_center(cut, 1000)
        librosa.output.write_wav('./tmp/single/single.wav', pad, sr = 1000)
        RFC_Predict = RFC_Single()
        RFC_Digits = RFC_Predict
        CNN_Predict = CNN_Single()
        CNN_Digits = CNN_Predict
        os.remove('./tmp/single/single.wav')
        remaining = digits['Speech_1kHz'][0][(peak[0] + 500):len(digits['Speech_1kHz'][0])]
        counter -=1
        if len(remaining) == 0:
            counter = 0
        while counter > 0:
            cut = pd.DataFrame(remaining).apply(remaining_cut)
            if str(cut[0]) == 'Nothing to cut':
                break
            pad = librosa.util.pad_center(cut, 1000)
            librosa.output.write_wav('./tmp/single/single.wav', pad, sr = 1000)
            RFC_Predict = RFC_Single()
            CNN_Predict = CNN_Single()
            RFC_Digits = RFC_Digits + RFC_Predict
            CNN_Digits = CNN_Digits + CNN_Predict
            os.remove('./tmp/single/single.wav')
            remaining = remaining[1000:len(remaining)]
            counter -=1
        os.remove(app.config['UPLOAD_FOLDER'] + '/' + audio_Filename)
    return render_template('base.html', Audio_Filename = audio_Filename, Google_Digits = Google_Predict, RFC_Digits = RFC_Digits, CNN_Digits = CNN_Digits)