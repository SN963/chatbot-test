import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model

model = load_model('chatbot_model.h5')
import json
import random
from tkinter import *

# from googletrans import Translator
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
import pyttsx3

engine = pyttsx3.init()

# Language in which you want to convert
language = 'en'


# Passing the text and language to the engine,
# here we have marked slow=False. Which tells
# the module that the converted audio should
# have a high speed
from flask import Flask

app = Flask(__name__)


@app.route('/bot',methods=['GET'])

def hello():
    #sentence = "كيفيه التقديم فى الجامعه"
    sentence=input("Enter your question")
    #sentence = "How to apply"
    def clean_up_sentence(sentence):
        # tokenize the pattern - split words into array
        result = translator.translate(sentence)
        if result.src == 'ar':
            reshaped_text = arabic_reshaper.reshape(sentence)
            reshaped_statement = reshaped_text[::-1]
            trans = translator.translate(reshaped_statement, src='ar', dest='en')
            sentence_words = nltk.word_tokenize(trans.text)
            """
            result = translator.translate(sentence, src='ar', dest='en')
            sentence_words = nltk.word_tokenize(result.text)
            print(sentence_words)
            """
            print("First Method", sentence_words)
        else:
            sentence_words = nltk.word_tokenize(sentence)

        # stem each word - create short form for word
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

    def bow(sentence, words, show_details=True):
        # tokenize the pattern
        sentence_rec = translator.translate(sentence, dest='ar')
        reshaped_text = arabic_reshaper.reshape(sentence_rec.text)
        reshaped_statement = reshaped_text[::-1]
        # print(reshaped_statement)
        if sentence_rec == 'ar':
            arabic_reshaper.reshape(sentence_rec.text)
            rec_text = reshaped_text[::-1]
            # rec_text = translator.translate(sentence, dest='en')
            sentence_words = clean_up_sentence(rec_text)
            # print(rec_text)
        else:
            if translator.detect(sentence) == 'ar':
                change = translator.translate(sentence, dest='en')
                sentence_words = clean_up_sentence(change)
            else:
                sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)

    def predict_class(sentence, model):  # ******
        # filter out predictions below a threshold

        p = bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        error_threshold = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list

    def getresponse(ints, intents_json):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                print(result)
                break
        return result

    def chatbot_response(msg):
        ints = predict_class(msg, model)
        res = getresponse(ints, intents)
        return res

    # Creating GUI with tkinter
    from googletrans import Translator
    import arabic_reshaper

    # make an object from translator
    translator = Translator()
    #
    receive = translator.translate(sentence)
    print(receive)
    # reshaped_text = arabic_reshaper.reshape(receive.text)
    if (receive.src == 'ar'):
        receive = translator.translate(sentence, dest='ar')
        reshaped_text = arabic_reshaper.reshape(receive.text)
        rec_text = reshaped_text[::-1]
        print("You: " + rec_text + '\n')
    else:
        print("You: " + receive.text + '\n')
        # make the object from translator detect the source of the enyered language
        FromUser = translator.translate(receive.text).src
        #

        res = chatbot_response(receive.text)
        # make the destination language same as the source
        result = translator.translate(res, dest=FromUser)
        print(result)

        if result.dest == 'ar':

            reshaped_text = arabic_reshaper.reshape(result.text)
            rev_text = reshaped_text[::-1]
            return "Bot: " + rev_text + '\n'
        else:
            return "Bot: " + result.text + '\n'


if __name__ == '__main__':
    app.run(debug=True)
