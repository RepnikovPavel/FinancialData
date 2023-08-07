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

import matplotlib
matplotlib.use('TkAgg')

formatter_ = {
    'index': {'transform': lambda x: x, 'to_type': 'same'},
    'acc_now_delinq': {'transform': lambda x: int(x), 'to_type': 'int'}, # real poss cat
    'addr_state': {'transform': lambda x: x, 'to_type': 'same'},          # cat
    'annual_inc': {'transform': lambda x: float(x), 'to_type': 'float'},     # real comparable
    'chargeoff_within_12_mths': {'transform': lambda x: int(x), 'to_type': 'int'}, # cat possible real
    'collections_12_mths_ex_med': {'transform': lambda x: int(x), 'to_type': 'int'}, # cat possible real
    'delinq_2yrs': {'transform': lambda x: int(x), 'to_type': 'int'}, # real comparable
    'dti': {'transform': lambda x: x, 'to_type': 'same'}, # real comparable 
    'earliest_cr_line': {'transform': lambda x: float(parse(x).year)+float(parse(x).month)/12.1,'to_type': 'float'}, # real 'Oct-1996' -> 1996.83  
    'emp_length': {'transform': lambda x: x, 'to_type':'same'}, # cat
    'fico_range_high': {'transform': lambda x: int(x) , 'to_type':'int'}, # real  linked  with fico_range_low (part of segment)
    'fico_range_low': {'transform': lambda x: int(x), 'to_type':'int'}, # real linked with fico_range_high (part of segment)
    'funded_amnt': {'transform': lambda x: float(x), 'to_type':'float'}, # real compatible  
    'home_ownership': {'transform': lambda x: x,'to_type': 'same'}, # cat
    'inq_last_12m': {'transform': lambda x: int(x), 'to_type':'int'}, # real compatible 
    'installment': {'transform': lambda x: float(x),'to_type':'float'}, # real relative(to amount?) compatible 
    'int_rate': {'transform': lambda x: int(np.round(float(x.replace('%','')))), 'to_type':'int'}, # real '8.99%' -> 9 
    'issue_d': {'transform': lambda x: float(parse(x).year)+float(parse(x).month)/12.1,'to_type':'float'}, # real 'Oct-1996' -> 1996.83, 
    'loan_amnt': {'transform': lambda x: float(x),'to_type':'float'}, # real comp
    'mort_acc': {'transform': lambda x: int(x), 'to_type':'int'}, # real comp 
    'mths_since_last_delinq': {'transform': lambda x: int(x), 'to_type':'int'}, # real comp 
    'mths_since_recent_bc_dlq': {'transform': lambda x: int(x), 'to_type':'int'}, # real comp 
    'mths_since_recent_inq': {'transform': lambda x: int(x), 'to_type':'int'}, # real comp 
    'num_accts_ever_120_pd': {'transform': lambda x: int(x), 'to_type':'int'}, # real poss comp
    'num_actv_bc_tl': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'num_rev_accts': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'num_sats': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'num_tl_120dpd_2m': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'num_tl_30dpd': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'num_tl_90g_dpd_24m': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'num_tl_op_past_12m': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'open_acc': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'open_il_24m': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'open_rv_24m': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'percent_bc_gt_75': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'pub_rec': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'pub_rec_bankruptcies': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'purpose': {'transform': lambda x: x,'to_type': 'same'}, # cat
    'revol_util': {'transform': lambda x: int(np.round(float(x.replace('%','')))), 'to_type':'int'},  # real  '8.99%' -> 9 
    'tax_liens': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'term': {'transform': lambda x: x,'to_type': 'same'}, # cat
    'title': {'transform': lambda x: x,'to_type': 'same'}, # cat external data link(bank names)
    'total_acc': {'transform': lambda x: int(x), 'to_type':'int'}, 
    'verification_status': {'transform': lambda x: x,'to_type': 'same'}, # cat
    'zip_code': {'transform': lambda x: x,'to_type': 'same'} # cat external data link(geo data) 
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

# cat_features_that_different_by_unique_values_in_train_and_test= [
#     'acc_now_delinq', # real poss cat
#     'addr_state',          # cat
#     'chargeoff_within_12_mths', # cat possible real
#     'collections_12_mths_ex_med', # cat possible real
#     'delinq_2yrs', # real comparable
#     'emp_length', # cat
#     'home_ownership', # cat
#     'inq_last_12m', # real compatible 
#     'mort_acc', # real comp 
#     'mths_since_last_delinq', # real comp 
#     'mths_since_recent_bc_dlq', # real comp 
#     'mths_since_recent_inq', # real comp 
#     'num_accts_ever_120_pd', # real poss comp
#     'num_actv_bc_tl', 
#     'num_rev_accts', 
#     'num_sats', 
#     'num_tl_120dpd_2m', 
#     'num_tl_30dpd', 
#     'num_tl_90g_dpd_24m', 
#     'num_tl_op_past_12m', 
#     'open_acc', 
#     'open_il_24m', 
#     'open_rv_24m', 
#     'percent_bc_gt_75', 
#     'pub_rec', 
#     'pub_rec_bankruptcies', 
#     'purpose', # cat
#     'tax_liens', 
#     'term', # cat
#     'title', # cat external data link(bank names)
#     'total_acc', 
#     'verification_status', # cat
#     'zip_code' # cat external data link(geo data) 
# ]

# cat_features_ = [
#     'acc_now_delinq', # real poss cat
#     'addr_state',          # cat
#     'chargeoff_within_12_mths', # cat possible real
#     'collections_12_mths_ex_med', # cat possible real
#     'delinq_2yrs', # real comparable
#     'emp_length', # cat
#     'home_ownership', # cat
#     'inq_last_12m', # real compatible 
#     'mort_acc', # real comp 
#     'mths_since_last_delinq', # real comp 
#     'mths_since_recent_bc_dlq', # real comp 
#     'mths_since_recent_inq', # real comp 
#     'num_accts_ever_120_pd', # real poss comp
#     'num_actv_bc_tl', 
#     'num_rev_accts', 
#     'num_sats', 
#     'num_tl_120dpd_2m', 
#     'num_tl_30dpd', 
#     'num_tl_90g_dpd_24m', 
#     'num_tl_op_past_12m', 
#     'open_acc', 
#     'open_il_24m', 
#     'open_rv_24m', 
#     'percent_bc_gt_75', 
#     'pub_rec', 
#     'pub_rec_bankruptcies', 
#     'purpose', # cat
#     'tax_liens', 
#     'term', # cat
#     'title', # cat external data link(bank names)
#     'total_acc', 
#     'verification_status', # cat
#     'zip_code' # cat external data link(geo data) 
# ]


def reformat_content(table:pd.DataFrame,formatter_):
    o_ = table.copy()
    for i,column_name in enumerate(formatter_):
        print('{}% column name {}'.format(int(((i+1)/len(formatter_))*100),column_name))
        if column_name not in o_:
            print('WARNING: {} not in table'.format(column_name))
            continue
        pandas_type = ''
        if formatter_[column_name]['to_type'] == 'int':
            pandas_type = 'Int64'
        if formatter_[column_name]['to_type'] == 'float':
            pandas_type = 'float64'
        if formatter_[column_name]['to_type'] == 'same':
            continue
        o_[column_name] = o_[column_name].apply(lambda x: formatter_[column_name]['transform'](x) if pd.notnull(x) else x)
        o_[column_name] = o_[column_name].apply(pd.to_numeric, errors='coerce')
        o_ = o_.astype({column_name: pandas_type})
        # o_[column_name] = o_[column_name].astype(pandas_type)
    return o_

def X_train_or_test_to_1_format(inputpath_,outputpath_):
    X = pd.read_csv(inputpath_,index_col=False)
    X_reformated_ = reformat_content(X,formatter_)
    X_reformated_.to_csv(outputpath_,index=False)

@jit(nopython=True)
def get_value_by_segment(x, counts_,bins_):
    for i in range(1,len(bins_)-1):
        if x<= bins_[i] and x>= bins_[i-1]:
            return counts_[i-1]
    return 0.0

@jit(nopython=True)
def get_bin_index_by_value(x, bins_):
    if x < bins_[0]:
        return -1
    if x> bins_[-1]:
        return len(bins_)
    for i in range(1,len(bins_)-1):
        if x<= bins_[i] and x>= bins_[i-1]:
            return i-1

class Distrib1D:
    counts_: np.array
    bins_: np.array
    max_: np.float64
    a_: np.float64
    b_: np.float64
    std_: np.float64
    def __init__(self,table:pd.DataFrame, column_name:str):
        self.counts_, self.bins_ = np.histogram(table[column_name].dropna(), density=True)
        self.max_ = np.max(self.counts_)
        self.a_= self.bins_[0]
        self.b_= self.bins_[-1]
        self.std_ = np.nanstd(table[column_name])
    def __call__(self, x):
        return get_value_by_segment(x,self.counts_,self.bins_)
    def mutate_value(self,x):
        # get segment of value
        # if current_segment == -1:
        # if current_segment == len(self.bins_):
        if self.std_ < self.bins_[1]-self.bins_[0]:
            # move to nearest cells 
            current_segment = get_bin_index_by_value(x)
            distance = int(np.random.normal(loc=1,scale=len(self.bins_)/3))
            new_pos = current_segment+distance
            return np.maximum(0,self.bins_[0] + new_pos*(self.bins_[1]-self.bins_[0]))
        else:
            dtype_ = str(type(x))
            delta = np.random.normal(loc=0.0,scale=self.std_)
            o_ = x + delta
            if 'int' in dtype_:
                return np.maximum(0,int(o_))
            else:
                return np.maximum(0,o_)


def rho_between_features_vectorized(table, r_index, r_indexes_2, cName, distributions, buffer)->np.array:
    f1 = table[cName][r_index]
    f2 = table[cName][r_indexes_2].to_numpy()
    distrib = distributions[cName]
    d1_v = distrib(f1)
    # d2_vs = f2.apply(distrib)
    for i in range(len(buffer)):
        buffer[i] = distrib(f2[i])

    o_ = 0.5*np.absolute(f1-f2)/(distrib.b_-distrib.a_) + 0.5/distrib.max_*(np.absolute(buffer-d1_v))
    where_nan = np.argwhere(np.isnan(o_))
    o_[where_nan] = 1.0
    return o_
def rho_between_row_and_rows(table, r_index, r_indexes_2, distributions,names_of_columns, corr_m):
    N = len(names_of_columns)

    distances_for_features = np.zeros(shape=(N,len(r_indexes_2)))
    # buffer
    distribs_vs = np.zeros(shape=(len(r_indexes_2),))
    for i in range(N):
        distances_for_features[i] = rho_between_features_vectorized(table, r_index,r_indexes_2,names_of_columns[i],distributions,buffer=distribs_vs)

    # [index if feauture][index of another object]
    D1 = np.sum(distances_for_features,axis=0)/(2*N)
    # corr_ij_vec = corr.loc[]
    # where_nan = np.argwhere(np.isnan(rho_i_vec))
    # rho_i_vec[where_nan] = 1.0
    # D1 = np.sum(rho_i_vec)/(2*N)
    # D2 = 0.0
    # D2 = np.zeros(shape=(N,len(r_indexes_2)))
    D2 = np.zeros(shape=(len(r_indexes_2),))
    for i in range(N-1):
        for j in range(i+1, N):
            max_ = np.maximum(distances_for_features[i],distances_for_features[j])
            D2 += corr_m[i][j]*max_
    D2 = D2 / (N*(N-1)) 
    # return D1+D2
    o_ = D1+D2
    return o_





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



    return X,Y

def get_neig(X,Y, formatters_):
    # make distributions
    distributions = {} 
    for cName in formatters_:
        if cName not in X:
            continue
        expected_type = formatters_[cName]['to_type']
        if 'int' in expected_type or 'float' in expected_type:
            distr_ = Distrib1D(X,cName)
            distributions.update({cName:distr_})

    names_of_columns = list(distributions.keys())
    corr_mat = X.corr()
    K = 3
    N = 10
    index_of_row_neighbors = {}
    corr_m = np.absolute(corr_mat.loc[names_of_columns][names_of_columns].to_numpy())
    for i in range(X.shape[0]):
        print('{}/{}'.format(i,X.shape[0]))
        indexes = np.setdiff1d(np.random.randint(low=0,high=X.shape[0],size=N),[i])
        distances = rho_between_row_and_rows(X, r_index=i,r_indexes_2=indexes,distributions=distributions,corr_m=corr_m,names_of_columns=names_of_columns)

        argsort_by_distance = np.argsort(distances)
        k_neig_indexes = [indexes[el] for el in argsort_by_distance]
        # base = [X.loc[i][names_of_columns].values]
        # n_ith = [X.loc[k_neig_indexes[k]][names_of_columns].values for k in range(K)]
        # matrix_ = np.array(base+n_ith)
        # x=pd.DataFrame(matrix_,columns=names_of_columns)
        # x= x.to_html()
        # html_template = """
        # <html>
        # <head></head>
        # <body>
        # {}        
        # </body>
        # </html>
        # """.format(x)
        # tmp_file = open(os.path.join(conf.data_folder,"tmp_plotting.html"),'w')
        # tmp_file.write(html_template)
        # tmp_file.close()
        # raise SystemExit
        index_of_row_neighbors.update({i:k_neig_indexes})
    return index_of_row_neighbors



def train_augmentation(X:pd.DataFrame,Y:pd.DataFrame,formatters_):

    # make distributions
    distributions = {} 
    for cName in formatters_:
        if cName not in X:
            continue
        expected_type = formatters_[cName]['to_type']
        if 'int' in expected_type or 'float' in expected_type:
            distr_ = Distrib1D(X,cName)
            distributions.update({cName:distr_})


    # nan augmentation for categorial features
    X_aug = X.copy()
    Y_aug = Y.copy()
    N = X_aug.shape[0]
    rate_of_categorial_nan = 0.5
    for nan_aug_feature in cat_features_that_different_by_unique_values_in_train_and_test:
        if nan_aug_feature in X_aug:
            position_of_nan = np.random.randint(low=0,high=N,size=int(N*rate_of_categorial_nan))
            X_aug.loc[position_of_nan, nan_aug_feature] = pd.NA
            # X_aug[nan_aug_feature] = np.insert(X_aug[nan_aug_feature].values, position_of_nan, np.nan)

    print('mutate numeric values using distribution knowledge')
    # mutate numeric values using distribution knowledge
    X_aug2 = X.copy()
    Y_aug2 = Y.copy()
    for cName in distributions:
        print(cName)
        X_aug2[cName] = X_aug2[cName].apply(distributions[cName])

    # nan augmentation for numeric features
    X_aug3 = X.copy()
    Y_aug3 = Y.copy()
    rate_of_numeric_nan = 0.5
    for cName in distributions:
        position_of_nan = np.random.randint(low=0,high=N,size=int(N*rate_of_numeric_nan))
        X_aug3.loc[position_of_nan, cName] = pd.NA


    X_final = pd.concat([X,X_aug,X_aug2,X_aug3])
    Y_final = pd.concat([Y,Y_aug,Y_aug2,Y_aug3])

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
        replaced_unknown_by_nan = x.where(x.isin(self.sklearnEncoder.classes_),other=pd.NA)
        vs = replaced_unknown_by_nan.values
        # make transofm ignore nan
        new_vs = -1*np.ones(shape=(vs.size,),dtype=vs.dtype) 
        new_vs[~pd.isnull(vs)] = self.sklearnEncoder.transform(vs[~pd.isnull(vs)])
        o_ = pd.Series(new_vs)
        o_ = o_.where(o_ != -1,other=pd.NA)
        # new_vs[pd.isnull(vs)] = pd.NA
        return o_
    
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
    # # convert types train data
    # print('train reformatting')
    # X_train_or_test_to_1_format(inputpath_=conf.train_table,outputpath_=conf.X_train_reformated)
    # # convert types test data 
    # print('test reformatting')
    # X_train_or_test_to_1_format(inputpath_=conf.test_table,outputpath_=conf.X_test_reformated)

    # drop columns, make new features, make label encoders
    # X_train,Y_train = _1_format_to_train_dataset(inputpath_=conf.X_train_reformated,lables_path=conf.train_target)
    # make_encoders(X_train,output_path=conf.cat_encoders_path)

    print('train encoding')
    X_train,Y_train = _1_format_to_train_dataset(inputpath_=conf.X_train_reformated,lables_path=conf.train_target)
    # # # make train dataset
    X_train_dataset = encode(X_train,encoders=torch.load(conf.cat_encoders_path))
    print('train augmentation')
    X_train_dataset,Y_train = train_augmentation(X_train_dataset,Y_train,formatters_=formatter_)
    X_train_dataset.to_csv(conf.X_train_dataset,index=False)
    Y_train.to_csv(conf.Y_train_dataset,index=False)

    print('test encoding')
    # make test dataset
    X_test = _1_format_to_test_dataset(inputpath_=conf.X_test_reformated)
    X_test_dataset = encode(X_test,encoders=torch.load(conf.cat_encoders_path))
    X_test_dataset.to_csv(conf.X_test_dataset,index=False)
    pass