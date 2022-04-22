import random
from pydoc import render_doc
import re
from unittest import result
from flask import Flask, render_template_string
from flask import render_template, request,Response
import tkinter as tk
from tkinter import filedialog
import sys
import csv
import string
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from keras.preprocessing.image import ImageDataGenerator ,array_to_img,image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import cv2
from PIL import Image as im










# print(img)
# print(img.shape)
model = tf.keras.models.load_model('ASL_Model.h5')
print(model.summary())


i = "x.jpg"
img = image.load_img(i, target_size=(28,28))
img_array = image.img_to_array(img)

img_batch = np.expand_dims(img_array, axis=-1)

print(img_batch.shape)
predict = model.predict(img_batch)
data = im.fromarray(predict)
print(data)
