# top of the file

# from cProfile import label
# from crypt import methods
import random
from pydoc import render_doc
import re
from unittest import result
from flask import Flask, render_template_string
from flask import render_template, request,Response
import tkinter as tk
from tkinter import filedialog
from kerastuner import RandomSearch
import sys
import csv
import string
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from matplotlib.figure import Figure
from keras.preprocessing.image import ImageDataGenerator ,array_to_img
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from sklearn.metrics import accuracy_score,confusion_matrix

from keras.callbacks import ReduceLROnPlateau
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img


app = Flask(__name__)



TRAINING_FILE = 'datas/sign_mnist_train.csv'
VALIDATION_FILE = 'datas/sign_mnist_test.csv'

train_data = pd.read_csv(TRAINING_FILE)
train_data.head()
test_data = pd.read_csv(VALIDATION_FILE)
test_data.head()
print("gello")

y_train = train_data['label']
y_test = test_data['label']
del train_data['label']
del test_data['label']

unique_labels = y_train.unique()
unique_labels = np.sort(unique_labels)

# y_train = train_data['label']
# y_test = test_data['label']
# del train_data['label']
# del test_data['label']


# ##parse Input Data
# def parse_data_from_input(x):
#     with open(x) as file:
#         reader = csv.reader(file, delimiter=',')    
#         imgs = []
#         labels = []
#         next(reader, None)
#         for row in reader:
#             label = row[0]
#             data = row[1:]
#             img = np.array(data).reshape((28, 28))

#             imgs.append(img)
#             labels.append(label)

#     images = np.array(imgs).astype(float)
#     labels = np.array(labels).astype(float)
#     return images, labels

# training_images, training_labels = parse_data_from_input(TRAINING_FILE)
# validation_images, validation_labels = parse_data_from_input(VALIDATION_FILE)


def changeToAlphabets(x):
    if(x==0):
        val = "a"
    elif(x==1):
        val = "b"
    elif(x==2):
        val = "c"
    elif(x==3):
        val = "d"
    elif(x==4):
        val = "e"
    elif(x==5):
        val ="f"
    elif(x==6):
        val="g"
    elif(x==7):
        val="h"
    elif(x==8):
        val="i"
    elif(x==9):
        val="j"
    elif(x==10):
        val="k"
    return val

##Display Input Images
def plot_categories(training_images, training_labels):
    fig, axes = plt.subplots(3, 10, figsize=(16, 15))
    axes = axes.flatten()
    letters = list(string.ascii_lowercase)

    for k in range(30):
        img = training_images[k]
        img = np.expand_dims(img, axis=-1)
        img = array_to_img(img)
        ax = axes[k]
        ax.imshow(img, cmap="Greys_r")
        ax.set_title(f"{letters[int(training_labels[k])]}")
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

def createFig():
    
    hist = train_data.label.hist(color='pink',bins=10)
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    # xs = range(100)
    # ys = [random.randint(1, 50) for x in xs]
    axis.hist(train_data.label,bins = 10)
    return fig




def preprocess_image(x):
    x = x/255
    x = x.reshape(-1,28,28,1) # convertin it into 28 x 28 gray scaled image
    return x

train_x = preprocess_image(train_data.values)
test_x = preprocess_image(test_data.values)




##Training Generator and Validation Generator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False) 














#Create Model
def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')])
  

    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
   
  
    return model
# def build_model():  
#     model = keras.Sequential([
#     keras.layers.Conv2D(
#         #filters=hp.Int('conv_1_filter', min_value=75, max_value=200, step=25),
#         kernel_size=(3,3),
#         activation='relu',
        
#         input_shape=(28,28,1)
#     ),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same'),
#     keras.layers.Conv2D(
#        # filters=hp.Int('conv_2_filter', min_value=50, max_value=125, step=25),
#         kernel_size=(3,3),
#         activation='relu',
#     ),
    
#     keras.layers.Dropout(
#        # rate = hp.Choice('drop_1_rate', values = [0.1,0.5])
#     ),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same'),
    
    
#     keras.layers.Conv2D(
#        # filters=hp.Int('conv_3_filter', min_value=25, max_value=75, step=25),
#         kernel_size=(3,3),
#         activation='relu',
#     ),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='same'),  
#     keras.layers.Flatten(),
#     keras.layers.Dense(
#        # units=hp.Int('dense_1_units', min_value=128, max_value=1024, step=32),
#         activation='relu'
#     ),
#     # keras.layers.Dropout(
#     #     rate = hp.Choice('drop_2_rate', values = [0.1,0.3])
#     # ),
#     keras.layers.Dense(24, activation='softmax')
#   ])
  
#     model.compile(optimizer=keras.optimizers.Adam(),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
  
#     return model

# tuner_search=RandomSearch(build_model,
#                           objective='val_accuracy',
#                           max_trials=5,directory='output',project_name="ASLdetection1")

# tuner_search.search(train_x,y_train,epochs=2,validation_data = (test_x, y_test))



model=create_model()

lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 3, verbose=1,factor=0.5, min_lr=0.00001)
##APP Routing Starts From Here

@app.route('/')
def hello_world():
    return render_template('home.html')
# def p():
#     print("Hello World")

@app.route('/read', methods=["POST"])
def home():
    n = request.form['nl'] 
    print(n)
    with open("datas/sign_mnist_train.csv", "r") as f:
        result = f.readlines()
    return render_template("read.html", result=[result[0],result[int(n)],n])

# @app.route('/parse')
# def parse():
    
#     # training_images, training_labels = parse_data_from_input(TRAINING_FILE)
#     # validation_images, validation_labels = parse_data_from_input(VALIDATION_FILE)
#     return render_template("parse.html", result = [training_images.shape,training_labels.shape,validation_images.shape,validation_labels.shape])


@app.route('/display')
# def view():
def plot_png():
    # training_images, training_labels = parse_data_from_input(TRAINING_FILE)
    return render_template("display.html",result = "hii")

@app.route('/head')
def head():
   
    outputrow = train_data.head()
    outputcolumn = train_data.columns
    count = train_data.label.value_counts
    
    fig = createFig()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(),mimetype='image/png')
   
    
   




# @app.route('/trainGen', methods=["POST"])
# def trainGen():
#     batchSize = request.form['n'] 
#     print(batchSize)
#     train_generator, validation_generator = train_val_generators(training_images, training_labels, validation_images, validation_labels,int(batchSize))
    
    
#     return render_template("home.html", result=train_generator.x.shape)



@app.route('/modelfit', methods = ['POST'])
def modelFit():
    nE = request.form['e']
    datagen.fit(train_x)
    # Train our model
    history = model.fit(datagen.flow(train_x,y_train, batch_size = 128) 
                    ,epochs = int(nE)
                    , validation_data = (test_x, y_test)
                    , callbacks = [lr_reduction])
    model.save("ASL_Model.h5")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    accuracy = createPlot(epochs,acc,loss)
    output = io.BytesIO()
    return render_template("modelfit.html",result = [epochs,acc,val_acc[-1],loss,val_loss])
    # loss = createPlot(epochs,loss)
    # output1 = io.BytesIO()

    
   # FigureCanvas(accuracy).print_png(output)

    # FigureCanvas(loss).print_png(output1)
   # return Response(output.getvalue(),mimetype='image/png'),render_template('modelfit.html',result="1")
    # return render_template("modelfit.html",result = Response(output.getvalue(),mimetype='image/png'))

   


def createPlot(epchos,accuracy,loss):
    train_df = pd.read_csv(TRAINING_FILE)
    hist = train_df.label.hist(color='pink',bins=10)
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    # xs = range(100)
    # ys = [random.randint(1, 50) for x in xs]
    axis.plot(epchos,accuracy,'r',label = 'Training Accuracy')
    axis.plot(epchos,loss,'b',label = 'Loss Accuracy')
    return fig

def predictions_to_labels(pred):
    labels =[]
    for p in pred:
        labels.append(unique_labels[int(np.argmax(p)) - 1])
    return labels


def changeToAlphabets(x):
    # val = "z"
    if(x==0):
        val = "a"
    elif(x==1):
        val = "b"
    elif(x==2):
        val = "c"
    elif(x==3):
        val = "d"
    elif(x==4):
        val = "e"
    elif(x==5):
        val ="f"
    elif(x==6):
        val="g"
    elif(x==7):
        val="h"
    elif(x==8):
        val="i"
    elif(x==9):
        val="j"
    elif(x==10):
        val="k"
    return val

@app.route('/predict', methods = ['POST'])
def predict():
    label = request.form['label']
    print(label)
    modelx = tf.keras.models.load_model('ASL_Model.h5')
    predictions = model.predict(test_x)
    print(predictions)
    y_pred_labels = predictions_to_labels(predictions)
    print(y_pred_labels)
   

    y_test_labels = predictions_to_labels(y_test)
    print(y_test_labels)
    x = accuracy_score(y_test_labels,y_pred_labels)
    return render_template("predict.html",result=[y_pred_labels[int(label)],y_test_labels[int(label)],x])


@app.route('/test', methods = ['POST'])
def test():
    imglbl = request.form['x']
    print(imglbl)
    modelx = tf.keras.models.load_model('Accurate.h5')
    predictions = modelx.predict(test_x)
    print(predictions)
    y_pred_labels = predictions_to_labels(predictions)
    y_test_labels = predictions_to_labels(y_test)
    x = accuracy_score(y_test_labels,y_pred_labels)
    return render_template("test.html",result=[y_pred_labels[int(imglbl)],y_test_labels(int[imglbl]),x])







    





    
    


    

    


   









if __name__ == '__main__':
  app.run(debug=True)
   


   
