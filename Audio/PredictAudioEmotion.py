#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import librosa # to extract speech features
import pickle
from tensorflow.keras.models import model_from_json


# In[3]:


with open("Audio/models/scaler.p", 'rb') as pickled:
        data = pickle.load(pickled)

mean = data['mean']
std = data['std']


# In[4]:



with open("Audio/models/model.json", "r") as json_file:   #Loading the saved model
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("Audio/models/Emotion_Model.h5")
loaded_model._make_predict_function()


# In[5]:


def predaudioemotion(wav_file_name):
    label_to_text = {0:'angry',1:'neutral', 2:'happy', 3:'sad'}
    y, sr = librosa.load(wav_file_name)
    data = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    data = (data - mean)/std
    data = np.asarray(data)
    data=data.reshape(1, -1)
    data = np.expand_dims(data, axis=2)
    return label_to_text[np.argmax(loaded_model.predict(data))]


# In[7]:


def findpitch(file):
    y, sr = librosa.load(file)
    stft = np.abs(librosa.stft(y))
    pitches, magnitudes = librosa.piptrack(y, sr=sr, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    # pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitch=[p for p in pitch if p>0]
    # pitchmean = np.mean(pitch)
#     pitchstd = np.std(pitch)
#     pitchmax = np.max(pitch)
    pitchmin = np.min(pitch)
    return pitchmin

