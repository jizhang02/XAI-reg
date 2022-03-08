import warnings
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import keras.backend as K
import cv2
import keras
import keras.models
import pandas as pd
from PIL import Image,ImageChops
import innvestigate
import innvestigate.utils as iutils
from innvestigate.tools import Perturbation, PerturbationAnalysis
from keras.models import load_model
from model_rebuilt import *
from utils import plot_image_grid
def huber_loss(y_true, y_pred, clip_delta=0.5):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta
  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
  return tf.where(cond, squared_loss, linear_loss)

def huber_loss_mean(y_true, y_pred, clip_delta=0.005):
  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))

# percent_mean_absolute_error
def pmae(y_true, y_pred):
  if not K.is_tensor(y_pred):
    y_pred = K.constant(y_pred)
  y_true = K.cast(y_true, y_pred.dtype)
  diff = K.mean(K.abs((y_pred - y_true))/K.mean(K.clip(K.abs(y_true),K.epsilon(),None)))
  return 100. * K.mean(diff)
def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.keras.backend.mean(tf.where(cond, squared_loss, linear_loss))


custom_objects={
'huber_loss':huber_loss,
'pmae':pmae
}

from tqdm import tqdm # progress bar
import glob

def dataloader(images_path, type=3):
  imagelist = sorted(glob.glob(os.path.join(images_path, '*.png')))
  if(type==3):X = np.zeros((int(len(imagelist)), 128,128,3))# for reading rgb images
  else: X = np.zeros((int(len(imagelist)), 128,128,1))
  for i in tqdm(range(len(imagelist))):
      if(type==3):img = plt.imread(imagelist[i]) # for reading rgb images
      else: img = Image.open(imagelist[i]).convert('L')
      img = np.array(img)
      # normalize
      img_mean = np.mean(img)
      img_std = np.std(img)
      img_norm = (1/img_std)*(img-img_mean)
      if(type==3):X[i] = img_norm   # for reading rgb images
      else: X[i] = img_norm[:, :, np.newaxis]

  return X

path = "X_test/"
x_test = dataloader(path,type=3)
y_test_csv = pd.read_csv('X_test/test_annotationcv01_128.csv')
y_test_px = y_test_csv['head circumference (mm)']/y_test_csv['pixel size(mm)']
HC_max = np.max(y_test_px)
y_test = y_test_px/HC_max# normalization
generator = iutils.BatchSequence([x_test, y_test], batch_size=16)
# Load a model
input_size = (128,128,3)

#model = resnet50(input_shape=input_size)#vgg16(input_shape=input_size)
#model.load_weights('models/model1.h5')
#note,resnet50HL have problem on decovnet, add twice on deconvet, like selected_methods_indices = [0,1,2,2,3,4,5,6,7]

model = load_model('models/resnet50MSE.h5',custom_objects)#vgg16MAE.h5 resnet50MAE
perturbation_function = "gaussian"#(0,0.3)"zeros" "mean" "invert"
region_shape  = (32, 32)# fit to the input image size (128,128)
steps = 15 # Number of perturbation steps.
regions_per_step = 1  # Perturbate 1 region per step

# Scale to [0, 1] range for plotting.

input_range = [-1, 1]
noise_scale = (input_range[1]-input_range[0]) * 0.1
ri = input_range[0]  # reference input


# Configure analysis methods and properties
methods = [
    # NAME                   OPT.PARAMS                 TITLE
    # Show input, Returns the input as analysis.
    #("input",                {},                        "Input"),
    # Returns the Gaussian noise as analysis.
   # ("random",               {},                        "Random"),
    # Function
    ("gradient",             {"postprocess": "abs"},    "Gradient"),
    ("smoothgrad",           {"noise_scale": noise_scale, "postprocess": "square"}, "SmoothGrad"),
    # Signal
    ("deconvnet",            {},                       "Deconvnet"),
    ("guided_backprop",      {},                       "Guided Backprop",),
    #("pattern.net",          {"pattern_type": "relu"}, "PatternNet"),
    # Interaction
   # ("pattern.attribution",  {"pattern_type": "relu"}, "PatternAttribution"),
    ("deep_taylor.bounded",  {"low": input_range[0],   "high": input_range[1]}, "DeepTaylor"),
    ("input_t_gradient",     {},                       "Input * Gradient"),
    ("integrated_gradients", {"reference_inputs": ri}, "Integrated Gradients"),
   # ("deep_lift.wrapper",    {"reference_inputs": ri}, "DeepLIFT Wrapper - Rescale"),
    #("deep_lift.wrapper",    {"reference_inputs": ri,  "nonlinear_mode": "reveal_cancel"}, "DeepLIFT Wrapper - RevealCancel"),
    ("lrp.z",                {},                       "LRP-Z"),
   # ("lrp.epsilon",          {"epsilon": 1},           "LRP-Epsilon"),
]

# Select methods of your choice
selected_methods_indices = [0,1,2,3,4,5,6,7]
#selected_methods_indices = [13,10,9,8,5,4,3,2]

#selected_methods_indices = [0]
selected_methods = [methods[i] for i in selected_methods_indices]
print('Using method(s) "{}".'.format([method[0] for method in selected_methods]))

# instantiate the analyzer objects
analyzers = [innvestigate.create_analyzer(method[0],  model, **method[1]) for method in selected_methods]

scores_selected_methods = dict()
perturbation_analyses = list()
for method, analyzer in zip(selected_methods, analyzers):
    print("Method: {}".format(method[0]))
    try:
        # Set up the perturbation analysis
        # This is the method with which the pixels in the most important regions are perturbated
        # Gaussian(mean=0.0,standard diviation=0.3)
        perturbation = Perturbation(perturbation_function, region_shape=region_shape, in_place=False)
        # Comment out to invert the perturbation order
        # perturbation.aggregation_function = lambda x, axis: -np.mean(x, axis=axis)
        perturbation_analysis = PerturbationAnalysis(analyzer, model, generator, perturbation, recompute_analysis=False,
                                                 steps=steps, regions_per_step=regions_per_step, verbose=True)

        test_loss = perturbation_analysis.compute_perturbation_analysis()# Scalar test loss (if the model has a single output and no metrics)
        #print(len(test_loss)) # the number of perturbation areas 16
        print(test_loss)
        # Store the scores and perturbation analyses for later use
        scores = np.array(test_loss)*HC_max # multiply with max y_test # one column
        Error = scores[:,1]
        AOPC = Error[0]-np.mean(Error)
        print("ERROR:",Error)
        print("AOPC:",AOPC)
        scores_selected_methods[method[0]] = np.array(scores)
        perturbation_analyses.append(perturbation_analysis)
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print('\n' + message)
        continue

#Plot the perturbation curves and compute area over the perturbation curve (AOPC) with baseline
# fig = plt.figure(figsize=(15, 5))
# aopc = list()  # Area over the perturbation curve
# baseline_accuracy = scores_selected_methods["random"][:, 1]
# for method_name in scores_selected_methods.keys():
#     scores = scores_selected_methods[method_name] # the shape is (16,3),(number of perturbations,, number of channels)
#     accuracy = scores[:, 1]
#     aopc.append(accuracy[0] - np.mean(accuracy))# AOPC of each analyser
#
#     label = "{} (AOPC: {:.3f})".format(method_name, aopc[-1])
#     plt.plot(accuracy - baseline_accuracy, label=label)


#Plot the perturbation curves and compute area over the perturbation curve (AOPC) without baseline
fig = plt.figure(figsize=(7.5, 5))
aopc = list()  # Area over the perturbation curve
for method_name in scores_selected_methods.keys():
    scores = scores_selected_methods[method_name] # the shape is (16,3),(number of perturbations,, number of channels)
    accuracy = scores[:, 1]#the second column, totally 3 columns
    aopc.append(accuracy[0] - np.mean(accuracy))# AOPC of each analyser

    label = "{} (AOPC: {:.3f})".format(method_name, aopc[-1])
    plt.plot(accuracy, label=label)

plt.xlabel("Perturbation steps")
plt.ylabel("Predicted ERROR of analyzers (pixels)")
#plt.xticks(np.array(range(scores.shape[0])))
plt.xticks(np.array(range(16)))

plt.legend()
plt.savefig('analysis_results/' + 'aopc' + ".pdf", bbox_inches='tight', pad_inches=0.0)
plt.show()

# # Now plot the perturbation step by step.
# for perturbation_analysis, method in zip(perturbation_analyses, selected_methods):
#     samples = list()
#
#     # Reset the perturbation_analysis
#     perturbation_analysis.perturbation.num_perturbed_regions = 1
#
#     sample = np.copy(x_test[0:1])# (1, 128, 128, 3)
#     analysis = perturbation_analysis.analyzer.analyze(sample)
#     a = analysis
#     a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
#     a /= np.max(np.abs(a))
#     b = np.squeeze(a)
#     plt.imshow(b, cmap="seismic", clim=(-1, 1))
#     plt.axis('off')
#     plt.savefig('analysis_results/' + 'analysis_' + str(method[0]) + ".png", bbox_inches='tight', pad_inches=0.0)
#     #aggregated_regions = perturbation_analysis.perturbation.reduce_function(np.moveaxis(analysis, 3, 1), axis=1,keepdims=True)
#     #aggregated_regions = perturbation_analysis.perturbation.aggregate_regions(aggregated_regions)
#     #ranks = perturbation_analysis.perturbation.compute_region_ordering(aggregated_regions)
#     #print(np.shape(ranks)) # (1, 1, 4, 4)
#     #print(np.shape(analysis))# (1, 128, 128, 3)
#     # Perturbate for some steps
#     for i in range(steps + 1):
#         # Plot the original image and analysis without any perturbation
#         if i > 0:
#             perturbation_analysis.perturbation.num_perturbed_regions += perturbation_analysis.regions_per_step
#             # Perturbate
#             sample = perturbation_analysis.compute_on_batch(sample, analysis)
#
#         a = sample
#         #a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
#         a /= np.max(np.abs(a))
#         b = np.squeeze(a)
#         #plt.imshow(b, cmap="binary", clim=(-1, 1)) # seismic binary
#         plt.imshow(b, cmap="binary", clim=(-1, 1)) # seismic binary
#         plt.axis('off')
#         plt.savefig('analysis_results/' + 'step_' + str(i) + str(method[0]) + ".png",bbox_inches='tight', pad_inches=0.0)