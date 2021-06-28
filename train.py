import os
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import adam

train_nomask_path = './Face Mask Dataset/Train/WithoutMask'
validation_nomask_path ='./Face Mask Dataset/Validation/WithoutMask'
test_nomask_path = './Face Mask Dataset/Test/WithoutMask'

train_mask_path = './Face Mask Dataset/Train/WithMask'
validation_mask_path = './Face Mask Dataset/Validation/WithMask'
test_mask_path = './Face Mask Dataset/Test/WithMask'



train_nomask_imges = os.listdir(train_nomask_path)
train_mask_imges = os.listdir(train_mask_path)

# Train Data (mask + no mask)
train_img = np.float32(np.zeros((4000, 224, 224, 3)))
train_label = np.float64(np.zeros((4000, 1)))

train_cnt = 0
for nomask_img in train_nomask_imges:
    img_full_path = os.path.join(train_nomask_path, nomask_img)
    img = load_img(img_full_path, target_size=(224, 224))

    im = img_to_array(img)
    im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    train_img[train_cnt, :, :, :] = im

    train_label[train_cnt] = 0
    train_cnt = train_cnt + 1
    if train_cnt == 2000:
        break

for mask_img in train_mask_imges:
    img_full_path = os.path.join(train_mask_path, mask_img)
    img = load_img(img_full_path, target_size=(224, 224))

    im = img_to_array(img)
    im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    train_img[train_cnt, :, :, :] = im

    train_label[train_cnt] = 1
    train_cnt = train_cnt + 1
    if train_cnt == 4000:
        break


# Test Data (mask + no mask)

test_nomask_imges = os.listdir(test_nomask_path)
test_mask_imges = os.listdir(test_mask_path)

test_nomask_len = len(test_nomask_imges)
test_mask_len = len(test_mask_imges)

test_img = np.float32(np.zeros((test_nomask_len + test_mask_len, 224, 224, 3)))
test_label = np.float64(np.zeros((test_nomask_len + test_mask_len, 1)))

test_cnt = 0
for nomask_img in test_nomask_imges:
    img_full_path = os.path.join(test_nomask_path, nomask_img)
    img = load_img(img_full_path, target_size=(224, 224))

    im = img_to_array(img)
    im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    test_img[test_cnt, :, :, :] = im

    test_label[test_cnt] = 0
    test_cnt = test_cnt + 1

for mask_img in test_mask_imges:
    img_full_path = os.path.join(test_mask_path, mask_img)
    img = load_img(img_full_path, target_size=(224, 224))

    im = img_to_array(img)
    im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    test_img[test_cnt, :, :, :] = im

    test_label[test_cnt] = 1
    test_cnt = test_cnt + 1

# 데이터셋 섞기(현재 00000 ...... 111111)
n = train_label.shape[0]
m = test_label.shape[0]

n_suffled_index = np.random.choice(n, size=n, replace=False)
m_suffled_index = np.random.choice(m, size=m, replace=False)

train_label = train_label[n_suffled_index]
train_img = train_img[n_suffled_index]

test_label = test_label[m_suffled_index]
test_img = test_img[m_suffled_index]


# mobilenet 전이학습
base_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
base_model.trainable = False
base_model.summary()
print("Number of layers in the base model: ", len(base_model.layers))

flatten_layer = Flatten()
dense_layer1 = Dense(128, activation='relu')
bn_layer1 = BatchNormalization()
dense_layer2 = Dense(1, activation=tf.nn.sigmoid)

model = Sequential([
    base_model,
    flatten_layer,
    dense_layer1,
    bn_layer1,
    dense_layer2,
])

base_learning_rate = 0.001
model.compile(optimizer=adam(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(train_img, train_label, epochs=5, batch_size=16, validation_data=(test_img, test_label))
model.save("model.h5")

