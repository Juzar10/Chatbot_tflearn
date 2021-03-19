import nltk
from nltk.stem import lancaster
from nltk.stem.lancaster import LancasterStemmer
from numpy.matrixlib.defmatrix import matrix

steamer = LancasterStemmer()

import numpy as np
# from tensorflow import keras
import tflearn
import tensorflow
import random
import json
import pickle


with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle" , "rb") as f:
        words,tags,training_input,training_output=pickle.load(f)

except:
    words = []
    tags  = []
    docs_x  = []
    docs_y  = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrd = nltk.word_tokenize(pattern)
            words.extend(wrd)
            docs_x.append(pattern)
            docs_y.append(intent["tag"])

        if intent["tag"] not in tags:
            tags.append(intent["tag"])


    words = [steamer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    tags = sorted(tags)

    training_input = []
    training_output = []

    tag_0_list = [0 for _ in tags]

    for x,doc in enumerate(docs_x):

        bag = []
        doc = nltk.word_tokenize(doc)
        wrds = [steamer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = tag_0_list[:]
        output_row[tags.index(docs_y[x])] = 1

        training_input.append(bag)
        training_output.append(output_row)


    training_input = np.array(training_input)
    training_output = np.array(training_output)

    with open("data.pickle" , "wb") as f:
        pickle.dump((words,tags,training_input,training_output) , f)



net = tflearn.input_data(shape= [None ,len(training_input[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(training_output[0]) , activation="softmax")

net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("chatbot.tflearn")
except:
    model.fit(training_input,training_output,n_epoch=1000,batch_size=8,show_metric=True)
    model.save("chatbot.tflearn")


def bag_of_words(s,words):

    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [steamer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("bot: Start talking")
    while True:
        print("You : " , end=" ")
        inp = input()
        if inp.lower == "quit":
            break
        
        result = model.predict([bag_of_words(inp,words)])[0]
        print(result)
        result_index = np.argmax(result)
        tag = tags[result_index]
        
        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("I didn't get that, try some other question")

chat()