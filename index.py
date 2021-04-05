from flask import Flask, redirect, url_for, render_template, request

from itertools import permutations
import numpy as np  
import pandas as pd
import os

import io
import random

import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras import backend as K


import chatbot as cb
import json

app = Flask(__name__)

# Home

@app.route('/')
def home():
    return render_template("index.html", content = "Home")


# Exercise 1


@app.route('/exercise_1', methods = ["GET"])
def exercise_1():
    return render_template("exercise_1.html")

@app.route('/exercise_1_result', methods = ["GET"])
def exercise_1_result():
    ans = einsteinSofution()
    result = "<h1>Теоретично са възможни четири решения, като последното (четвърто) е вярно:</h1>"
    result += ans
    return ans

def einsteinSofution():
    color       = ["бяла", "червена", "синя", "зелена", "жълта"]
    nationality = ["германец", "руснак", "японец", "американец", "англичанин"]
    animal      = ["котка", "чинчила", "куче", "костенурка", "рибка"]
    car         = ["опел", "нисан", "БМВ", "фолцваген", "мерцедес"]
    drink       = ["бира", "спрайт", "фанта", "пепси", "кока кола"]

    result = ""

    count = 0
    for p in permutations(nationality[1:]):
        p = ("германец", *p)                    #10
        for c in permutations(color):

            if (abs(c.index("зелена") - 
                c.index("бяла"))==1 and                 #6
                c[p.index("англичанин")]=="червена"       #2
                and abs(p.index("германец") - 
                c.index("синя"))==1):                   #15

                for l in permutations(animal):
                    if l[p.index("американец")]=="рибка":     #3
                        for d in permutations(car):
                            if (l[d.index("мерцедес")]=="куче" and  #7
                                c[d.index("нисан")]=="жълта"    #8
                                and abs(d.index("БМВ") - 
                                l.index("куче"))==1           #11
                                and abs(d.index("нисан") - 
                                l.index("котка"))==1 and               #12
                                d[p.index("японец")]=="фолцваген"): #14

                                     for e in permutations(drink[1:]):
                                        e = (*e[:2], "бира", *e[2:])  #9
                                        if (c[e.index("фанта")]=="зелена"#4
                                            and e[p.index("руснак")]=="спрайт"  #5
                                            and e[d.index("опел")]=="пепси"):                  #13

                                            #Долното  да се появи едва след като се натисне бутон
                                            count += 1
                                            #print("Solution {}:".format(count))
                                            result += "<div>" + "Решение {}:".format(count) + "</div>"
                                            #print("The {} uses Vim".format(p[e.index("Vim")]))
                                            result += "<div>" + "{} пие кока кола".format(p[e.index("кока кола")]) + "</div>"
                                            #print("The {} writes in C++".format(p[l.index("C++")]))
                                            result += "<div>" + "{} е собственик на чинчила".format(p[l.index("чинчила")]) + "</div>"
                                            print("\nПълното решение:")
                                            result += "<div>" + "Пълното решение:" + "</div>"
                                            for i in range(5):
                                                result += p[i] + "  |  " + c[i] + "  |  " + l[i] + "  |  " + d[i] + "  |  " + e[i] + "</br>"
                                            #print(p[i], c[i], l[i], d[i], e[i])
                                            result += "</br>"
    return result


# Exercise 2

@app.route('/exercise_2', methods = ["GET", "POST"])
def exercise_2():
    if request.method == "GET":
        return render_template("exercise_2.html", showAnswer = False)
    else:
        ans = mnistt()
        return render_template("exercise_2.html", showAnswer = True, answer = ans)

def mnistt():
    batch_size = 128
    num_classes = 10
    epochs = 1
    # input image dimensions
    img_rows, img_cols = 28, 28
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    for i in range(9):  
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
        plt.show()
  
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    result = ""
    #result += '<div>' + 'x_train shape:' + str(x_train.shape) + '</div>'

    #result += '<div>' + str(x_train.shape[0]) + 'train samples' + '</div>'
    #result += '<div>' + str(x_test.shape[1]) + 'train samples' + '</div>'

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
    model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)


    layer_outputs = [layer.output for layer in model.layers[1:7]]
    activation_model = Model(inputs=model.input,outputs=layer_outputs)

    img = x_test[52].reshape(1,28,28,1)
    activations = activation_model.predict(img)
    plt.imshow(x_test[52], cmap=plt.get_cmap('gray'))
    plt.show()

    layer_names = []
    for layer in model.layers[1:3]: layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
    images_per_row = 8
    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
    
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,:, :, col * images_per_row + row]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size: (row + 1) * size] = channel_image
                scale = 1. / size
                plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(display_grid, aspect='auto', cmap='viridis')
                plt.show()

    #result += '<div>' + 'Загуба:' + str(score[0]) + '</div>'
    result += '<div>' + 'Точност при класификация:' + str(score[1]*100) + '</div>'
    result += '<div>' + "</p>" + '</div>'
    #model_json = model.to_json()
    return result


# Exercise 23

@app.route('/exercise_3', methods = ["GET", "POST"])
def exercise_3():
    if request.method == "GET":
        return render_template("exercise_3.html", showAnswer = False)
    else:
        input = request.form['fname']
        resp = cb.getBotResponse(input)
        
        data = []

        if request.form.get('allmessages'):
            allMessagesJson = request.form['allmessages']
            data = json.loads(allMessagesJson)
            
        data.append({"text": input, "side": 'right'})
        data.append({"text": resp, "side": 'left'})
        
        messages = []
        for d in data:
            messages.append(cb.Message(d['text'], d['side']))

        allmessagesDump = json.dumps(data)
        return render_template("exercise_3.html", showAnswer = True, messages = messages, allmessages = allmessagesDump)


if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=8080)