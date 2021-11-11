import h5py
import numpy as np
import keras
import os
import glob
from skimage import io, color, exposure, transform
import matplotlib.pyplot as plt
from keras.models import model_from_json
from utils import preprocess_img

#defining classes and size of image used for training
num_classes = 43
size = 64

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

main_path = './GTSRB_Challenge/test/'
imgs = []
labels = []

# reading all the images from the dataset using glob for paths
all_img_paths = glob.glob(os.path.join(main_path, '*/*.ppm'))
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    img_path = str(img_path.replace("\\", "/"))
    print ("Processing image: " + str(img_path))
    img = preprocess_img(io.imread(img_path), size)
    label = get_label(img_path)
    imgs.append(img)
    labels.append(label)

#creating training numpy arrays
testX = np.array(imgs, dtype='float32')
testY = np.eye(num_classes, dtype='uint8')[labels]

testX = np.array(testX)
testY = np.array(testY)

# predict and evaluate
y_pred = loaded_model.predict_classes(testX)
acc = float(np.sum(y_pred == testY)) / np.size(y_pred)
print("Test accuracy = {}".format(acc))