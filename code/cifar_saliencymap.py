'''
-----------------------------------------------
File Name: cifar_saliencymap$
Description:
Author: Jing$
Date: 11/8/2021$
-----------------------------------------------
'''

#########################################################################
#########################################################################
from random import randint
import matplotlib.pylab as plt
import numpy as np
from keras.models import load_model
from keras.models import Sequential, Model
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K

import cv2
# load and prepate cifar images
model = load_model('cifarcnn.h5')
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
xtest = xtest[:10]
ytest = ytest[:10]
x_test = xtest.astype('float32')
x_test /= 255
nb_classes = 10
y_test = np_utils.to_categorical(ytest, nb_classes)


print("Getting feature maps of specific layers")
def get_feature_maps(model, layer_id, input_image):
    model_ = Model(inputs=[model.input],
                   outputs=[model.layers[layer_id].output])
    return model_.predict(np.expand_dims(input_image, axis=0))[0,:,:,:].transpose((2,0,1))

def plot_features_map(img_idx=None, layer_idx=[2, 4, 6, 8, 10, 12,14, 16],
                      x_test=x_test, ytest=ytest, cnn=model):
    if img_idx == None:
        img_idx = randint(0, ytest.shape[0]-1)
    input_image = x_test[img_idx]
    fig, ax = plt.subplots(3,3,figsize=(10,10))
    ax[0][0].imshow(input_image)
    ax[0][0].set_title('original img id {} - {}'.format(img_idx, labels[ytest[img_idx][0]]),fontsize=14)
    for i, l in enumerate(layer_idx):
        feature_map = get_feature_maps(cnn, l, input_image)
        print(feature_map.shape)
        ax[(i+1)//3][(i+1)%3].imshow(feature_map[0,:,:])
        ax[(i+1)//3][(i+1)%3].set_title('layer {} - {}'.format(l, cnn.layers[l].get_config()['name']),fontsize=14)
    #plt.savefig("cifarfeaturemap.pdf", dpi=200, orientation='landscape', format='pdf')
    plt.show()

    return img_idx


labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#plot_features_map()


# pip install scipy==1.1.0
from vis.visualization import visualize_saliency
print("Plotting saliency maps...")

def plot_saliency(img_idx=None):
    img_idx = plot_features_map(img_idx)
    grads = visualize_saliency(model, -1, filter_indices=ytest[img_idx][0],
                               seed_input=x_test[img_idx], backprop_modifier=None,
                               grad_modifier="absolute")
    predicted_label = labels[np.argmax(model.predict(x_test[img_idx].reshape(1,32,32,3)),1)[0]]
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(x_test[img_idx])
    ax[0].set_title('original img id {} - {}'.format(img_idx, labels[ytest[img_idx][0]]),fontsize=14)
    ax[1].imshow(grads, cmap='jet')
    ax[1].set_title('saliency - predicted {}'.format(predicted_label),fontsize=14)
    plt.savefig("cifarsaliency.pdf", dpi=200, orientation='landscape', format='pdf')
    plt.show()

plot_saliency()