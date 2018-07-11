import os
from random import shuffle

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np
import cv2
import matplotlib.pyplot as plt


from utils import config_utils
from utils import data_utils
from processors import *
from processors.master_preprocessor import MasterPreprocessor
from nn import *
from nn.nn_dispatcher import NN_DISPATCHER, NN_PARAMS


#constants for image labels for better readability
POSITIVE = 1
NEGATIVE = 0

# loading config params from file
config = config_utils.get_config(os.path.dirname(__file__), "config/config.json")

# fetch labeled images
pos_labeled_images = data_utils.load_images_with_labels(config["pos_images_path"],
	POSITIVE, os.path.dirname(__file__))

neg_labeled_images = data_utils.load_images_with_labels(config["neg_images_path"],
	NEGATIVE, os.path.dirname(__file__))

print("Loaded {} positive label images and {} negative label images"
	.format(len(pos_labeled_images), len(neg_labeled_images)))

# preprocess labeled images
labeled_images = pos_labeled_images + neg_labeled_images
master_preprocessor = MasterPreprocessor(config)
labeled_images = MasterPreprocessor(config).preprocess_image_list(labeled_images)

print("Preprocessed ({}) {} images".format(config["preprocessors"], len(labeled_images)))

# randomize and split training and test sets
shuffle(labeled_images)

X, y = zip(*labeled_images)
y_binary = to_categorical(y)
X = np.array(X)
X = X.astype("float")/255.0

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.33, random_state=42)

print("Train/Test split: {} train, {} test".format(len(X_train), len(X_test)))

print("Compiling network...")
neural_net_container = NN_DISPATCHER[config["nn"]](config)
model = neural_net_container.build()
opt = NN_PARAMS[config["opt"]](config["learning_rate"])
model.compile(loss=config["loss_function"], optimizer=opt, metrics=config["metrics"])

print("Training network...")
H = model.fit(X_train, y_train, 
	validation_data=(X_test, y_test), 
	batch_size=config["batch_size"], 
	epochs=config["epochs"], verbose=config["verbose"])

print("Evaluating network results...")
predictions = model.predict(X_test, batch_size=config["batch_size"])

print(classification_report(y_test.argmax(axis=1),
predictions.argmax(axis=1),
target_names=config["target_names"]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


