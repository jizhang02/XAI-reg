'''
-----------------------------------------------
File Name: Feature_maps$
Description: draw feature maps from models according to specific layers
Author: Jing$
Date: 11/15/2021$
-----------------------------------------------
'''

#########################################################################
#########################################################################
from random import randint
import matplotlib.pylab as plt
import numpy as np
from keras.models import load_model
from keras.models import Sequential, Model
from keras import backend as K
from vis.visualization import visualize_saliency

import cv2
def IOU(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection ) / (K.sum(y_true_f) + K.sum(y_pred_f) )

def dice_coef_loss(y_true, y_pred):
    smooth = 0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def Dice_loss_smooth(smooth = 1):
     return dice_coef_loss

def Kappa_new(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    a = K.sum(y_true_f * y_pred_f)
    b = K.sum((1-y_pred_f)*y_true_f)
    c = K.sum((1-y_true_f)*y_pred_f)
    d = K.sum((1-y_pred_f)*(1-y_true_f))
    kappa = 2.*(a*d-b*c)/((a+b)*(b+d)+(a+c)*(c+d))
    return 1-kappa

def Kappa_loss(y_true, y_pred):
    smooth=1
    Gi = K.flatten(y_true)
    Pi = K.flatten(y_pred)
    N = 256.0*256.0
    numerator = 2*K.sum(Pi * Gi)-K.sum(Pi)*K.sum(Gi)/N
    denominator = K.sum(Pi)+K.sum(Gi)-2*K.sum(Pi*Gi)/N
    return  (1 - (numerator+smooth)/(denominator+smooth))
def Kappa_loss_smooth(smooth = 1):
    return Kappa_loss

def Tversky(y_true, y_pred):  # with tensorflow
    alpha = 0.5
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return 1 - (true_pos) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos)

def Tversky_loss(alpha=0.5):
    return Tversky
custom_objects = {
     'Dice_loss_smooth': Dice_loss_smooth(smooth = 0),
     'dice_coef_loss':dice_coef_loss,
     'Kappa_loss':Kappa_loss,
     'Kappa_new':Kappa_new,
     'Tversky':Tversky,
     'Tversky_loss': Tversky_loss(alpha=0.5),
     'Kappa_loss_smooth':Kappa_loss_smooth(smooth=1),
     'IOU':IOU
     }

# load images

def load_data(img):
    input_img = cv2.imread(img)
    test_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.resize(test_img,(256,256))
    # normalize
    img_mean = np.mean(test_img)
    img_std = np.std(test_img)
    img_norm = (1 / img_std) * (test_img - img_mean)
    #x = img_norm[np.newaxis, :, :, np.newaxis]
    #test_img = x
    img_norm = img_norm[:,:,np.newaxis]
    print(img_norm.shape)
    test_img = img_norm
    return input_img, test_img


def get_feature_maps(model, layer_id, input_image):
    model_ = Model(inputs=[model.input],outputs=[model.layers[layer_id].output])
    # expand_dims: Insert a new axis, corresponding to a given position in the array shape.
    return model_.predict(np.expand_dims(input_image, axis=0))[0,:,:,:].transpose((2,0,1))
'''
color maps:
Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, 
BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, 
Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r,
PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, 
Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, 
PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, 
RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, 
Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, 
Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, 
YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, 
autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, 
bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, 
copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, 
gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, 
gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, 
gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, 
gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, 
jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, 
ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, 
rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, 
summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, 
tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, 
twilight_r, twilight_shifted, 
twilight_shifted_r, viridis, viridis_r, winter, winter_r
'''
def plot_features_map(ori_img,layer_idx=[8, 13, 18, 24, 32, 39, 49,53], x_test=None, cnn=None):

    fig, ax = plt.subplots(3,3,figsize=(10,10))
    ax[0][0].imshow(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))# show original color
    ax[0][0].set_title('original img',fontsize=14)
    for i, layer in enumerate(layer_idx):
        feature_map = get_feature_maps(cnn, layer, x_test)
        print(feature_map.shape)# attention the shape order
        ax[(i+1)//3][(i+1)%3].imshow(feature_map[0,:,:],cmap='jet')#,cmap='Greens'
        ax[(i+1)//3][(i+1)%3].set_title('layer {} - {}'.format(layer, cnn.layers[layer].get_config()['name']),fontsize=14)

    plt.savefig("ftmap_dice.pdf", dpi=200, orientation='landscape', format='pdf')
    plt.show()

# load model
model = load_model("ISIC2016Dice.h5", custom_objects)# ISIC2016Kappa.h5
model.summary()
conf = model.get_config()
#print(conf)
print(model.layers[1])
original_img,test_img = load_data("197.png")
plot_features_map(ori_img = original_img,x_test=test_img,cnn=model)
