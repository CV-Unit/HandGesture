# coding: utf-8
import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras import optimizers

GESTURE_FOLDER = "GestureKind"
MODEL_NAME = "samples_model.h5"
LABEL_NAME = "samples_labels.dat"

data = []
labels = []
class_num = 0
train_num = 20

try:
    os.makedirs(GESTURE_FOLDER)
except OSError as e:
    print(GESTURE_FOLDER+'文件夹已创建')

print('请将手势图片种类放在目录下的GestureKind文件夹中')
class_num = int(input('请输入手势种类数：'))
train_num = int(input('请输入训练次数：'))

for image_file in paths.list_images(GESTURE_FOLDER):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (100,100))
    image = np.expand_dims(image, axis=2)
    label = image_file.split(os.path.sep)[-2]
    data.append(image)
    labels.append(label)
print('数据标签加载完成')

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.2, random_state=0)

lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

with open(LABEL_NAME, "wb") as f:
    pickle.dump(lb, f)
print('生成dat文件，开始构建神经网络')

model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same",
                 input_shape=(100,100, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(class_num, activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.RMSprop(lr=0.0001),
              metrics=["accuracy"])

print('卷积神经网络构建成功，开始训练')
history = model.fit(X_train, Y_train,
                    validation_data=(X_test,Y_test), batch_size=32,
                    epochs=train_num, verbose=1)

model.save(MODEL_NAME)

print('训练完成，h5文件保存完成')


