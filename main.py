import random

import PIL.Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askopenfile
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


import distutils

if distutils.version.LooseVersion(tf.__version__) <= '2.0':
    raise Exception(
        'This notebook is compatible with TensorFlow 1.14 or higher, for TensorFlow 1.13 or lower please use the previous version at https://github.com/tensorflow/tpu/blob/r1.13/tools/colab/fashion_mnist.ipynb')


def convertJPGToMinst(im):
    try:
        im = im.resize((28, 28))
        im = im.convert('L')
        im = np.array(im)
        im = -im
        im = im.reshape(1, 28, 28, 1)
        im = im / 255.0
        return im
    except IOError:
        pass


def trainModel():
    model = keras.Sequential(
        [keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(128, activation=tf.nn.relu),
         keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    return model


def predictImageFromJPG(img, model):
    img = convertJPGToMinst(img)
    predictions = model.predict([img])
    prediction = predictions[0]
    label = np.argmax(prediction)

    print(prediction)
    print(prediction[label] * 100)
    print(class_names[label])
    return prediction


def predictImageFromMnist(img, model):
    img = (np.expand_dims(img, 0))
    predictions = model.predict([img])
    prediction = predictions[0]
    label = np.argmax(prediction)

    print(prediction)
    print(prediction[label] * 100)
    print(class_names[label])
    return prediction


def showMnistImage(img):
    plt.imshow(img)
    plt.show()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()


def getJPGImage(image):
    try:
        # im = Image.open(jpgImageName)
        im = image.resize((28, 28))
        im = im.convert('L')
        im = np.array(im)
        im = -im
        return im
    except IOError:
        pass


def showHighestPredictions(predictions):
    n = 3
    arr = predictions.argsort()[-n:][::-1]
    print("1.", class_names[arr[0]], " ", predictions[arr[0]] * 100, "%")
    print("2.", class_names[arr[1]], " ", predictions[arr[1]] * 100, "%")
    print("3.", class_names[arr[2]], " ", predictions[arr[2]] * 100, "%")
    dict = {
        class_names[arr[0]]: predictions[arr[0]] * 100,
        class_names[arr[1]]: predictions[arr[1]] * 100,
        class_names[arr[2]]: predictions[arr[2]] * 100
    }
    return dict


# --------------------------------------main------------------------------------

class_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
               'Ankle boot']
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
model = trainModel()

# ---------------------------GUI----------------------------


root = tk.Tk()
canvas = tk.Canvas(root, width=1200, height=800)
canvas.grid(columnspan=2, rowspan=2)


def load_JPG():
    file = askopenfile(parent=root, mode='rb', title="Wybierz zdjecie", filetype=[("JPG file", "*.jpg")])
    if file:
        text_box = tk.Label(root, text="                     ", font=("Raleway", 25))
        print("Dziala")
        im = PIL.Image.open(file)
        img = getJPGImage(im)
        figure = plt.figure(figsize=(5, 5), dpi=100)
        plt.axis("off")
        figure.add_subplot(111).imshow(img, cmap=plt.cm.binary)
        chart = FigureCanvasTkAgg(figure, root)
        chart.get_tk_widget().grid(row=0, column=0)
        # predykcja
        predictions = predictImageFromJPG(im, model)
        showHighestPredictions(predictions)
        # wypisanie wyniku
        text = tk.StringVar()
        drawChart(predictions)

        text.set(class_names[np.argmax(predictions)])
        text_box = tk.Label(root, text=class_names[np.argmax(predictions)], width=20, font=("Raleway", 25))
        text_box.grid(column=0, row=1, sticky="n")


def load_MNIST():
    n = random.randint(0, 10000)
    img = test_images[n]
    figure = plt.figure(figsize=(5, 5), dpi=100)
    plt.axis("off")
    figure.add_subplot(111).imshow(img, cmap=plt.cm.binary)
    chart = FigureCanvasTkAgg(figure, root)
    chart.draw()
    chart.get_tk_widget().grid(row=0, column=0)
    # predykcja
    predictions = predictImageFromMnist(img, model)
    showHighestPredictions(predictions)
    # zrobienie wykresu
    drawChart(predictions)
    # wypisanie wyniku
    predicted_label = np.argmax(predictions)
    true_label = test_labels[n]
    if predicted_label == true_label:
        text_box = tk.Label(root, text=class_names[np.argmax(predictions)], width=20, font=("Raleway", 25), fg="green")
    else:
        correct_label = "" + str(class_names[np.argmax(predictions)]) + "(" + str(class_names[true_label]) + ")"
        text_box = tk.Label(root, text=correct_label, width=20, font=("Raleway", 25), fg="red")
    # text_box = tk.Label(root, text=class_names[np.argmax(predictions)],width=20, font=("Raleway", 25))
    text_box.grid(column=0, row=1, sticky="n")


def drawChart(predictions):
    frame = tk.Frame(root)
    frame.grid(row=0, column=1)
    dict = showHighestPredictions(predictions)
    figure2 = plt.figure(figsize=(5, 5), dpi=100)
    #plt.axis("off")
    plt.xlabel("Label")
    plt.ylabel("Prediction %")
    figure2.add_subplot(111).bar(list(dict.keys()), dict.values(), color='g')
    chart = FigureCanvasTkAgg(figure2, frame)
    chart.draw()
    chart.get_tk_widget().grid(row=0, column=1)

def _quit():
    root.quit()
    root.destroy()



frame = Frame(root)
frame.grid(column=0, row=1)
# browse button JPG
browse_text = tk.StringVar()
jpg_btn = tk.Button(root, textvariable=browse_text, command=lambda: load_JPG(), font="Raleway", bg="#20bebe",
                    fg="white", height=2, width=30)
browse_text.set("JPG")
jpg_btn.grid(column=0, row=1, sticky='e')

# mnist button
browse_text = tk.StringVar()
mnist_btn = tk.Button(root, textvariable=browse_text, command=lambda: load_MNIST(), font="Raleway", bg="#20bebe",
                      fg="white", height=2, width=30)
browse_text.set("MNIST")
mnist_btn.grid(column=0, row=1, sticky='w')




button = Button(master=root, text="Quit", command=_quit, font="Raleway", bg="#D0312D", fg="white", height=4, width=10)
button.grid(column=1, row=1)

root.mainloop()

