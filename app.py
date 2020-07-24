import sys
import os
import numpy as np
from functools import wraps
from predictemt import pred, removeout, vidframe, ssimscore1
from flask import Flask, request, render_template, flash, redirect, session, url_for
from record import start_AVrecording,stop_AVrecording,file_manager
from Audio.AudioTextAnalysis import *
from Audio.PredictAudioEmotion import *
from vid2aud import vidtoaud
from werkzeug.utils import secure_filename
from tensorflow.keras.models import model_from_json
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import keyboard
import random


facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   #load face detection cascade file

app = Flask(__name__,static_url_path="/static/", static_folder="static")
app.secret_key = 'some secret key'




@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		if 'file' in request.files:
			f = request.files['file']  #getting uploaded video 
			basepath = os.path.dirname(__file__)
			file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
			f.save(file_path)  #saving uploaded video

			result, face = vidframe(file_path) #running vidframe with the uploaded video
			print("Generated Facial Emotion")
			vidtoaud(file_path)
			print("Generating Audio")
			words,rate,confidence,positive,negative,neutral=gettextfromspeech("uploads/sample.wav")
			print("Calculated words and confidence")
			pitch=findpitch("uploads/sample.wav")
			print("pitch generated")
			emot=predaudioemotion("uploads/sample.wav")
			print("Audio Emotion Generated")
			#os.remove(file_path)  #removing the video as we dont need it anymore
			#os.remove("uploads/sample.wav")
		else:
			start_AVrecording()
			keyboard.wait("q")
			stop_AVrecording()
			file_manager()
			result, face = vidframe("output.avi")
			words,rate,positive,negative,neutral=gettextfromspeech("output.wav")
			pitch=findpitch("output.wav")
			emot=predaudioemotion("output.wav")
		try:
			smileindex=result.count('happy')/len(result)  #smileIndex
			smileindex=round(smileindex,2)
		except:
			smileindex=0

		ssimscore=[ssimscore1(i,j) for i, j in zip(face[: -1],face[1 :])]  # calculating similarityscore for images
		posture = np.mean(ssimscore)
		emocount=[result.count('happy'),result.count('sad'),result.count('anger'),result.count('disgust'),result.count('fear')]
		posran = [random.randint(1,10) for x in range(len(positive))]
		negran = [random.randint(1,10) for x in range(len(negative))]
		neuran = [random.randint(1,10) for x in range(len(neutral))]
		posnum = [i for i in range(len(posran))]
		negnum = [i for i in range(len(negran))]
		neunum = [i for i in range(len(neuran))]
		print(posran)

		# datainp = {"faceex":emocount, "posture":posture, "smileindex":int(smileindex),
  #       					"rate":int(rate),"confidence":confidence,"positive":positive,"negative":negative,"neutral":neutral,
  #       					"pitch":pitch, "emot":emot}
		return render_template("predict.html", faceex=emocount, posture = posture,smileindex=smileindex, rate=rate, confidence=confidence, positive=positive,
		negative=negative,neutral=neutral, pitch=pitch, emot=emot, posran=posran, negran=negran, neuran=neuran, posnum=posnum, negnum=negnum, neunum=neunum) #returning all the three variable that can be displayed in html
	return None

@app.route('/tryit')
def tryit():
    return render_template('tryit.html')


@app.route('/features')
def features():
    return render_template('features.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
    app.secret_key = 'some secret key'
