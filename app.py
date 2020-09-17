from flask import Flask, render_template, request,redirect, url_for
import tensorflow as tf
import keras.backend as k
import keras
from keras.models import Sequential, Model
from keras.layers import Bidirectional, LSTM, Dense, TimeDistributed, Embedding, RepeatVector, Input
from keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
import time
from keras.layers import Layer
from keras import utils
from keras.optimizers import RMSprop, adam, Adam
import time

global model,graph

#tf_config = some_custom_config
sess = tf.Session()
graph = tf.get_default_graph()

k.set_session(sess)
enc_model = keras.models.load_model('encoder.h5',compile=False)
dec_model = keras.models.load_model('decoder.h5',compile=False)

maxlen_ask = 19
maxlen_reply = 11

token = {}
try:
    with open('dict.txt', 'r') as f:
        for i in f:

            temp1 = i.split(',')
            for j in temp1:
                temp2 = j.split(':')
                var_one = temp2[0]
                var_two = int(temp2[1])
                token[var_one] = var_two
except:
    pass
def clean_text(text):
    text = text.lower()
    text = re.sub(r",", ",", text)
    text = re.sub(r"’", "'", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she will", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"why's", "why is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"aren't", "are not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r" ok ", " okay ", text)
    text = re.sub(r" thankyou", "thank you", text)
  

    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

def str_to_token(sentence:str):
    words = sentence.lower().split()
    token_list = []
    for word in words:
        token_list.append(token[word])
    return keras.preprocessing.sequence.pad_sequences([token_list], maxlen=maxlen_ask, padding='post')




app = Flask(__name__)


    


@app.route("/", methods=['GET','POST'])
def index():
    # if request.method == 'POST':
    #     return redirect(url_for('app'))
    # else:
    return render_template('app.html')


@app.route("/", methods=['GET','POST'])
def inp():
    if request.method == "POST":
        n = request.form['text']
        return n


@app.route("/predict", methods=['GET','POST'])
def show_value():
    temp_2 = inp()
    temp_2 = str(temp_2)
    temp_2 = clean_text(temp_2)
    # text = TextBlob(temp_2)
    # temp_2 = text.correct()

    ask=''
    input_user= temp_2
    split_input=input_user.split(' ')
    for i in range(len(split_input)):
        if split_input[i] not in token:
            #unk_word=split_input[i]
            split_input[i]= 'ukn'
        ask +=split_input[i]+' '
        with sess.as_default():
            with graph.as_default():
                #k.set_session(sess)
                state_value = enc_model.predict(str_to_token(ask))
        empty_target_seq = np.zeros((1,1))
        empty_target_seq[0,0] = token['sos']
        stop_condition =False
        decoded_translation = ''
        
        
        while not stop_condition:
            with sess.as_default():
                with graph.as_default():
                #k.set_session(sess)
                    dec_output, h, c = dec_model.predict([empty_target_seq]+state_value)
            sample_word_index = np.argmax(dec_output[0,-1,:])
                    ##print(dec_output[0,-1,:])
            sample_word = None
            for word, index in token.items():
                if sample_word_index == index:
                    decoded_translation += ' {}' .format(word)
                    sample_word = word
                        
            if sample_word == 'eos' or len(decoded_translation.split())> maxlen_reply:
                stop_condition = True

            empty_target_seq =np.zeros((1,1))
            empty_target_seq[0,0] = sample_word_index
            state_value = [h,c]
            hot = decoded_translation
                    
            hot = hot.replace('eos'," ")
            hot = 'Chatbot : '+hot
            user = 'You : ' +temp_2  
    return render_template('app.html',text1=hot,text=user)




if __name__ == "__main__":
    print('wait')
    app.run(debug=True)
