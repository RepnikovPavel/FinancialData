from dateutil.relativedelta import *
from dateutil.easter import *
from dateutil.rrule import *
from dateutil.parser import *
from datetime import *
from sklearn.preprocessing import LabelEncoder
from numba import jit
import conf
from UI.LOG import *
import cv2
from aml.train_pipeline import *  
from aml.train_pipeline import *
import pandas as pd

import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN_ResNet50_FPN_Weights
from torchinfo import summary

# from torchvision.models.detection import ssd300_vgg16,SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead,SSD300_VGG16_Weights,det_utils
from torchvision.models.detection import ssd300_vgg16

import torch

import aml.model_using as model_using
import aml.support_func as support_funcS
import aml.time_mesuarment as time_mesuarment
import sys

import aml.managers as managers
import aml.img_processing as img_processing
import random
import numpy as np
import pprint
from torchinfo import summary
from aml.img_processing import *

from PIL import Image
import aml.models as models
import matplotlib.pyplot as plt
from pprint import pprint as Print
from PIL import Image
import warnings
from torchvision.utils import draw_bounding_boxes  
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.ops import nms 
from torchmetrics.detection.mean_ap import MeanAveragePrecision as mAP
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from IPython.display import IFrame, display, HTML
import sklearn
from data_manip import Distrib1D,formatter_
from tqdm import tqdm
from sklearn import cluster
from sklearn import decomposition
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pyclustering.cluster import optics,dbscan,kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from UI.plotting import *
from pyclustering.cluster import cluster_visualizer
tqdm.pandas()
import matplotlib

def rho_between_features(f1,f2,distrib)->float:
    p1 = distrib(f1)
    p2 = distrib(f2)
    o_ = 0.5*np.absolute(f1-f2)/(distrib.b_-distrib.a_) + 0.5/distrib.max_*(np.absolute(p2-p1))
    return o_

def rho_(x_1,x_2, distributions,names_of_columns, corr_m):
    N = len(names_of_columns)
    rho_i  = np.zeros(shape=(N,))
    for i in range(N):
        rho_i[i] = rho_between_features(x_1[i],x_2[i],distributions[names_of_columns[i]])
    D1 = np.sum(rho_i)/(2*N)
    D2 = 0.0
    for i in range(N-1):
        for j in range(i+1, N):
            D2 += corr_m[i][j]*np.maximum(rho_i[i],rho_i[j])
    D2 = D2 / (N*(N-1)) 
    return D1+D2

def get_all_distances(X, metric):
    N_ = len(X)
    distances_ = np.zeros(shape=(int(N_*(N_-1)/2),))
    k_ = 0
    for i in range(N_-1):
        print('\r{}/{}'.format(i,N_-2),end='')
        for j in range(i+1,N_):
            distances_[k_] = metric(X[i],X[j])
            k_ += 1
    return distances_

X = pd.read_csv(conf.X_train_reformated,index_col=False)
Y = pd.read_csv(conf.train_target,index_col=False).drop(columns='index')


distributions = {} 
for cName in formatter_:
    if cName not in X:
        continue
    expected_type = formatter_[cName]['to_type']
    if 'int' in expected_type or 'float' in expected_type:
        distr_ = Distrib1D(X,cName)
        distributions.update({cName:distr_})
numeric_columns = [el for el in list(distributions.keys())]
X_numeric = X[numeric_columns]
for cName in X_numeric:
    if X_numeric[cName].notna().sum() < 0.8*X_numeric.shape[0]:
        X_numeric = X_numeric.drop(columns=[cName])
numeric_columns = [el for el in X_numeric]
X_numeric = X_numeric.dropna()
# X_for_plot_targets =  X_numeric.sample(n=100000)
# indexes_for_plot = X_for_plot_targets.index
# Y_for_plot_targets = Y.loc[indexes_for_plot].values.flatten()
# X_for_plot_targets = sklearn.preprocessing.Normalizer().fit_transform(X_for_plot_targets)
X_numeric = X_numeric.sample(n=400).to_numpy()


all_names=  [el for el in X]
to_drop = np.setdiff1d(all_names,numeric_columns)
corr_m = np.absolute(X.drop(columns=to_drop).corr().loc[numeric_columns][numeric_columns].to_numpy())
user_function = lambda point1, point2: rho_(point1,point2,distributions,numeric_columns,corr_m)
metric = distance_metric(type_metric.USER_DEFINED, func=user_function)
# distances = get_all_distances(X_numeric,metric)
# fig_,ax_ = plot_float_distribution(data=distances,fig_size=(16,9),title=str(np.percentile(distances,q=1)))
# plt.show()
# alg_instance = dbscan.dbscan(data=X_numeric,eps=0.02, neighbors=5,metric = metric)

alg_instance = cluster.OPTICS(metric=metric,min_samples=4,n_jobs=4)
alg_instance.fit(X_numeric)
labels_ = alg_instance.labels_
# labels_ = Y
N_ = len(np.unique(labels_))
pca_model = decomposition.PCA(n_components=3)
pca_model.fit(X_numeric)
NumericIn = pca_model.transform(X_numeric)

data_dict = {'x':NumericIn[:,0], 'y':NumericIn[:,1],'z':NumericIn[:,2],'label':labels_}
answers = pd.DataFrame(data=data_dict)
fig = plt.figure(figsize=(16,9))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
cmap = matplotlib.colors.ListedColormap(sns.color_palette("bright", N_).as_hex())
colors_ = [cmap(el) for el in labels_]
ax.scatter(answers['x'], answers['y'], answers['z'], c=colors_, marker='o', alpha=1) 




# labels_ = Y_for_plot_targets
# N_ = len(np.unique(labels_))
# pca_model = decomposition.PCA(n_components=3)
# pca_model.fit(X_for_plot_targets)
# NumericIn = pca_model.transform(X_for_plot_targets)

# data_dict = {'x':NumericIn[:,0], 'y':NumericIn[:,1],'z':NumericIn[:,2],'label':labels_}
# answers = pd.DataFrame(data=data_dict)
# fig = plt.figure(figsize=(16,9))
# ax = Axes3D(fig, auto_add_to_figure=False)
# fig.add_axes(ax)
# cmap = matplotlib.colors.ListedColormap(sns.color_palette("bright", N_).as_hex())
# colors_ = [cmap(el) for el in labels_]
# ax.scatter(answers['x'], answers['y'], answers['z'], c=colors_, marker='o', alpha=1) 

plt.show()



