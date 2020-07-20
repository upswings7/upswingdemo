#!/usr/bin/env python
# coding: utf-8

# In[1]:

import librosa

#dependency for speech-to-text
import io
import numpy as np
# from google.oauth2 import service_account
# credentials = service_account.Credentials.from_service_account_file('Audio/models/api_key.json')

#dependency for sentiment'
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
analyser = SentimentIntensityAnalyzer()
import speech_recognition as sr


# In[4]:


def gettextfromspeech(speech_file,language="en-US"):
    """Transcribe the given audio file synchronously and output the word time
    offsets."""
    # trans=[]             #list for storing transcript,words and confidence
    # textword=[]
    # confid=[]
    lengt = librosa.get_duration(filename=speech_file)

    # Initialize recognizer class (for recognizing the speech)
    r = sr.Recognizer()

    # Reading Audio file as source
    # listening the audio file and store in audio_text variable
    
    with sr.AudioFile(speech_file) as source:
        
        audio_text = r.listen(source)
        
    # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
        try:
            
            # using google speech recognition
            text = r.recognize_google(audio_text)
            print('Converting audio transcripts into text ...')
            
         
        except:
             print('Sorry.. run again...')
    
    # from google.cloud import speech
    # from google.cloud.speech import enums
    # from google.cloud.speech import types  

      
    # client = speech.SpeechClient(credentials=credentials)      #using google's api for speech to text

    # with io.open(speech_file, 'rb') as audio_file:
    #     content = audio_file.read()
              

    # audio = types.RecognitionAudio(content=content)

    # config = types.RecognitionConfig(
    #         language_code=language,
    #         enable_word_time_offsets=True)

    # response = client.recognize(config, audio) 

    # for result in response.results:
    #     alternative = result.alternatives[0]
    #     trans.append(alternative.transcript)
    #     confid.append(alternative.confidence)

        # for word_info in alternative.words:
        #     textword.append(word_info.word)
    
    

#Sentiment Analysis part

    tokens=nltk.word_tokenize(text)                 #Generating Tokens
    sprate=len(tokens)/lengt
    tokens=[token for token in tokens if len(token)>1]
    stop_words = set(stopwords.words('english'))    
    fwords = [w for w in tokens if not w in stop_words]       #Removing Stop words

    pos=[]
    neg=[]
    neu=[]
    for i in fwords:
        score = analyser.polarity_scores(i)
        if score["neu"]>score["neg"] and score["neu"]>score["pos"]:
            neu.append(i)
        elif score["pos"]>score["neg"] and score["pos"]>score["neu"]:
            pos.append(i)
        elif score["neg"]>score["pos"] and score["neg"]>score["neu"]:
            neg.append(i)
            
#             start_time = word_info.start_time
#             end_time = word_info.end_time
#             print('Word: {}, start_time: {}, end_time: {}'.format(
#                 word,
#                 start_time.seconds + start_time.nanos * 1e-9,
#                 end_time.seconds + end_time.nanos * 1e-9))
    return(fwords,sprate,pos,neg,neu)


# In[5]:


#words,SpeedRate,confidence,positive,negative,neutral=transcribe_file_with_word_time_offsets("sam.wav")


# In[ ]:




