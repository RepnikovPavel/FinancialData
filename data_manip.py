from dateutil.relativedelta import *
from dateutil.easter import *
from dateutil.rrule import *
from dateutil.parser import *
from datetime import *
from sklearn.preprocessing import LabelEncoder

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

formatter_ = {
    'index': lambda x: x,
    'acc_now_delinq': lambda x: int(x), # real poss cat
    'addr_state': lambda x: x,          # cat
    'annual_inc': lambda x: float(x),     # real comparable
    'chargeoff_within_12_mths': lambda x: int(x), # cat possible real
    'collections_12_mths_ex_med': lambda x: int(x), # cat possible real
    'delinq_2yrs': lambda x: int(x), # real comparable
    'dti': lambda x: x, # real comparable 
    'earliest_cr_line': lambda x: float(parse(x).year)+float(parse(x).month)/12.1, # real 'Oct-1996' -> 1996.83  
    'emp_length': lambda x: x, # cat
    'fico_range_high': lambda x: int(x), # real  linked  with fico_range_low (part of segment)
    'fico_range_low': lambda x: int(x), # real linked with fico_range_high (part of segment)
    'funded_amnt': lambda x: float(x), # real compatible  
    'home_ownership': lambda x: x, # cat
    'inq_last_12m': lambda x: int(x), # real compatible 
    'installment': lambda x: float(x), # real relative(to amount?) compatible 
    'int_rate': lambda x: int(np.round(float(x.replace('%','')))), # real '8.99%' -> 9 
    'issue_d': lambda x: float(parse(x).year)+float(parse(x).month)/12.1, # real 'Oct-1996' -> 1996.83, 
    'loan_amnt': lambda x: float(x), # real comp
    'mort_acc': lambda x: int(x), # real comp 
    'mths_since_last_delinq': lambda x: int(x), # real comp 
    'mths_since_recent_bc_dlq': lambda x: int(x), # real comp 
    'mths_since_recent_inq': lambda x: int(x), # real comp 
    'num_accts_ever_120_pd': lambda x: int(x), # real poss comp
    'num_actv_bc_tl': lambda x: int(x), 
    'num_rev_accts': lambda x: int(x), 
    'num_sats': lambda x: int(x), 
    'num_tl_120dpd_2m': lambda x: int(x), 
    'num_tl_30dpd': lambda x: int(x), 
    'num_tl_90g_dpd_24m': lambda x: int(x), 
    'num_tl_op_past_12m': lambda x: int(x), 
    'open_acc': lambda x: int(x), 
    'open_il_24m': lambda x: int(x), 
    'open_rv_24m': lambda x: int(x), 
    'percent_bc_gt_75': lambda x: int(x), 
    'pub_rec': lambda x: int(x), 
    'pub_rec_bankruptcies': lambda x: int(x), 
    'purpose': lambda x: x, # cat
    'revol_util': lambda x: int(np.round(float(x.replace('%','')))), # real  '8.99%' -> 9 
    'tax_liens': lambda x: int(x), 
    'term': lambda x: x, # cat
    'title': lambda x: x, # cat external data link(bank names)
    'total_acc': lambda x: int(x), 
    'verification_status': lambda x: x, # cat
    'zip_code': lambda x: x # cat external data link(geo data) 
}

cat_features_that_different_by_unique_values_in_train_and_test= [
    'zip_code',
    'title'
]

cat_features_ = [
    'addr_state',
    'emp_length',
    'home_ownership',
    'purpose',
    'term',
    'title',
    'verification_status',
    'zip_code' 
]


def reformat_content(table:pd.DataFrame,ColumnNameLambdaPair):
    for i,cName in enumerate(table):
        print('{}% column name {}'.format(int(((i+1)/table.shape[1])*100),cName))
        table[cName] = table[cName].apply(lambda x: ColumnNameLambdaPair[cName](x) if pd.notnull(x) else x)
    return table

def X_train_or_test_to_1_format(inputpath_,outputpath_):
    X = pd.read_csv(inputpath_,index_col=False)
    X_reformated_ = reformat_content(X,formatter_)
    X_reformated_.to_csv(outputpath_,index=False)

def _1_format_to_train_dataset(inputpath_:str,lables_path:str)-> pd.DataFrame:
    X = pd.read_csv(inputpath_,index_col=False)
    # gen new features
    x2 = X['fico_range_high']
    x1 = X['fico_range_low']
    tmp = (x1+x2)/2.0
    tmp = tmp.astype('int64')
    X['fico_range_mid'] = tmp
    tmp2 = x2-x1
    tmp2 = tmp2.astype('int64')
    X['fico_range_length'] = tmp2
    
    Y = pd.read_csv(lables_path,index_col=False).drop(columns=['index'])
    # remove 
    X.drop(columns=[
                    'index',
                    # 'zip_code',
                    'title',
                    'fico_range_high',
                    'fico_range_low',
                    # 'issue_d'
                    ],inplace=True)

    # augmentation
    X_aug = X.copy()
    Y_aug = Y.copy()
    N = X_aug.shape[0]
    rate_of_categorial_nan = 0.5
    for nan_aug_feature in cat_features_that_different_by_unique_values_in_train_and_test:
        if nan_aug_feature in X_aug:
            position_of_nan = np.random.randint(low=0,high=N,size=int(N*rate_of_categorial_nan))
            X_aug.loc[position_of_nan, nan_aug_feature] = np.nan
            # X_aug[nan_aug_feature] = np.insert(X_aug[nan_aug_feature].values, position_of_nan, np.nan)



    X_final = pd.concat([X,X_aug])
    Y_final = pd.concat([Y,Y_aug])

    return X_final,Y_final

def _1_format_to_test_dataset(inputpath_:str)-> pd.DataFrame:
    X = pd.read_csv(inputpath_,index_col=False)
    # gen new features
    x2 = X['fico_range_high']
    x1 = X['fico_range_low']
    tmp = (x1+x2)/2.0
    tmp = tmp.astype('int64')
    X['fico_range_mid'] = tmp
    tmp2 = x2-x1
    tmp2 = tmp2.astype('int64')
    X['fico_range_length'] = tmp2
    
    # remove 
    X.drop(columns=[
                    'index',
                    # 'zip_code',
                    'title',
                    'fico_range_high',
                    'fico_range_low',
                    # 'issue_d'
                    ],inplace=True)
    return X


class CategorialEncoder:
    sklearnEncoder: LabelEncoder
    def __init__(self,encoder:LabelEncoder) -> None:
        self.sklearnEncoder = encoder
    def transform(self,x: pd.Series):
        replaced_unknown_by_nan = x.where(x.isin(self.sklearnEncoder.classes_),other=np.nan)
        vs = replaced_unknown_by_nan.values
        # make transofm ignore nan
        new_vs = np.zeros(shape=(vs.size,),dtype=vs.dtype) 
        new_vs[~pd.isnull(vs)] = self.sklearnEncoder.transform(vs[~pd.isnull(vs)])
        new_vs[pd.isnull(vs)] = np.nan
        return pd.Series(new_vs)
    
def make_encoders(X:pd.DataFrame,output_path:str):
    cNames = [el for el in X]
    encoders = {}
    for i,cat_feature in enumerate(cat_features_):
        print('{}% column name {}'.format(int(((i+1)/len(cat_features_))*100),cat_feature))
        if cat_feature not in cNames:
            continue
        cData = X[cat_feature].dropna().to_list()
        enc = LabelEncoder()
        enc.fit(cData)
        encoders.update({cat_feature:CategorialEncoder(enc)})
    torch.save(encoders,output_path)

def encode(X:pd.DataFrame,encoders:dict[str,LabelEncoder])->pd.DataFrame:
    encoders_names=  list(encoders.keys())
    for i,feature in enumerate(X):
        print('{}% column name {}'.format(int(((i+1)/X.shape[1])*100),feature))
        if feature not in encoders_names:
            continue
        # if label exists in train and not exist in test -> BUG
        X[feature] = encoders[feature].transform(X[feature])
        X[feature] = X[feature].astype('category')
    return X

if __name__ == '__main__':
    # convert types
    X_train_or_test_to_1_format(inputpath_=conf.train_table,outputpath_=conf.X_train_reformated)

    # drop columns, make new features, make label encoders
    X_train,Y_train = _1_format_to_train_dataset(inputpath_=conf.X_train_reformated,lables_path=conf.train_target)
    make_encoders(X_train,output_path=conf.cat_encoders_path)

    print('train encoding')
    # # # make train dataset
    X_train,Y_train = _1_format_to_train_dataset(inputpath_=conf.X_train_reformated,lables_path=conf.train_target)
    X_train_dataset = encode(X_train,encoders=torch.load(conf.cat_encoders_path))
    X_train_dataset.to_csv(conf.X_train_dataset,index=False)
    Y_train.to_csv(conf.Y_train_dataset,index=False)

    # convert types test data 
    X_train_or_test_to_1_format(inputpath_=conf.test_table,outputpath_=conf.X_test_reformated)

    print('test encoding')
    # make test dataset
    X_test = _1_format_to_test_dataset(inputpath_=conf.X_test_reformated)
    X_test_dataset = encode(X_test,encoders=torch.load(conf.cat_encoders_path))
    X_test_dataset.to_csv(conf.X_test_dataset,index=False)
    pass