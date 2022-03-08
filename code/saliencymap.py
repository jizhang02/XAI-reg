# Explainable AI with iNNvestigate tool.
# Author: Jing Zhang
# Date: 2020-04-09
import os

import SimpleITK as sitk
import cv2
import innvestigate.utils
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#from efficientnet.keras import EfficientNetB2
from keras.applications.vgg16 import VGG16

from keras.models import load_model


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.keras.backend.mean(tf.where(cond, squared_loss, linear_loss))


def pmae(y_true, y_pred):  # percent_mean_absolute_error
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    diff = K.mean(K.abs((y_pred - y_true)) / K.mean(K.clip(K.abs(y_true), K.epsilon(), None)))
    return 100. * K.mean(diff)


custom_objects = {
    'huber_loss': huber_loss,
    'pmae': pmae
}


def load_image(path, type=2, size=(224, 224)):
    if type == 3:  # for RGB image
        image = cv2.imread(path)  # for RGB
        print(image.shape[0:2])
        if image.shape[0:2] != size:
            image = cv2.resize(image, size)
        # normalize
        img_mean = np.mean(image)
        img_std = np.std(image)
        img_norm = (1 / img_std) * (image - img_mean)
        x = img_norm[np.newaxis, :, :]  # for RGB
    else:  # for grayscale image
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # for grayscale
        if image.shape[0:2] != size:
            image = cv2.resize(image, size)
        # normalize
        img_mean = np.mean(image)
        img_std = np.std(image)
        img_norm = (1 / img_std) * (image - img_mean)
        x = img_norm[np.newaxis, :, :, np.newaxis]  # for grayscale
    return x


def load3d_img(img_path, W, H, slice):
    itk_img = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(itk_img)
    img_resize = np.zeros((img.shape[0], W, H), dtype=np.float32)

    for k in range(0, img.shape[0]):  # for each slice
        img_resize[k] = cv2.resize(img[k], (W, H), interpolation=cv2.INTER_AREA)  # resize
        img_resize[k] = (img_resize[k] - np.mean(img_resize[k])) / np.std(img_resize[k])  # normalize
    img_resize = img_resize.reshape(H, W, slice)
    x = img_resize[np.newaxis, :, :, :]
    print(x.shape)  # 1,224,224,9
    return x


# def image_reading(file_path):
#     g = os.walk(file_path)
#     for path, d, filelist in g:
#         for filename in filelist:
#             if filename.endswith('png'):
#                 print(os.path.join(path, filename))

if __name__ == "__main__":
    file_path = './acdc/'
    images = []
    models = []
    for filename in os.listdir(r"./" + file_path):
        # print(filename) #just for test
        if filename.endswith('gz'):
            img = load3d_img(file_path+ "/" + filename,224,224,9)
            images.append(img)
        # if filename.endswith('png'):
        #     img = load_image(file_path + "/" + filename, 3, (224, 224))
        #     images.append(img)

    # models.append(load_model('models/best_vgg.h5',custom_objects))
    models.append(load_model('models/best_regresnetmae.h5', custom_objects))
    # models.append(load_model('models/best_resnet.h5', custom_objects))
    # models.append(load_model('models/best_densenet.h5',custom_objects))
    # models.append(load_model('models/best_xception.h5', custom_objects))
    # models.append(load_model('models/best_mobile.h5', custom_objects))
    # models.append(load_model('models/best_inception.h5', custom_objects))

    # methods = [ 'gradient', 'smoothgrad', 'deconvnet',  'guided_backprop','deep_taylor', 'input_t_gradient', 'integrated_gradients',
    #            'lrp.z']
    methods = ['guided_backprop']
    # Compare images
    for im in range(len(images)):
        print("Image:", im)
        # Compare networks
        for m in range(len(models)):
            print("Model:", m)
            # Compare analyzers
            for i in range(len(methods)):
                print("Method:", i)
                try:
                    analyzer = innvestigate.create_analyzer(methods[i], models[m])
                    a = analyzer.analyze(images[im])  # analyze based predicted value
                    print(a.shape)
                    for k in range(0, a.shape[3]):  # for 3d slices
                        #a[:,:,:,k] = a[:,:,:,k].sum(axis=np.argmax(np.asarray(a[:,:,:,k].shape) == 3))# 返回0，所以按列找最大值，
                        a[:,:,:,k] = a[:,:,:,k].sum(axis=0)# axis=0 按列求和，
                        a[:,:,:,k] /= np.max(np.abs(a[:,:,:,k]))
                        b = np.squeeze(a[:,:,:,k])# 去掉维度是1的维度
                        plt.imshow(b, cmap="seismic", clim=(-1, 1))  # seismic
                        plt.axis('off')
                        plt.savefig('./' + 'img_' + str(k) + 'model' + str(m) + str(methods[i]) + ".png",
                                    bbox_inches='tight', pad_inches=0.0)

                    # a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3)) #for 2d images
                    # a /= np.max(np.abs(a))
                    # b = np.squeeze(a)
                    # plt.imshow(b, cmap="seismic", clim=(-1, 1))  # seismic
                    # plt.axis('off')
                    # plt.savefig('./' + 'img_' + str(m) + 'model' + str(m) + str(methods[i]) + ".png",
                    #             bbox_inches='tight', pad_inches=0.0)

                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print('\n' + message)
                    continue
