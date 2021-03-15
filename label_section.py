#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

def getcount(x):
    return x.count()
def getset(x):
    return len(set(x))

def LabelSection(label):
    'label区间特征'

    '用户领取优惠券的次数，种类'
    luc_cnt_set = pd.pivot_table(label,index='user_id',values='coupon_id',aggfunc=[getcount,getset]).reset_index()
    luc_cnt_set.columns = ['user_id','luc_cnt','luc_set']
    label = pd.merge(label,luc_cnt_set,on='user_id',how='left')

    '用户领取商家的次数，种类'
    lum_set = pd.pivot_table(label,index='user_id',values='merchant_id',aggfunc=[getcount,getset]).reset_index()
    lum_set.columns = ['user_id','lum_cnt','lum_set']
    label = pd.merge(label,lum_set,on='user_id',how='left')

    '用户领取距离的最大、最小、平均'
    lucdis_stat = pd.pivot_table(label,index='user_id',values='coupon_id',aggfunc=[np.max,np.min,np.mean]).reset_index()
    lucdis_stat.columns = ['user_id','luc_dismax','luc_dismin','luc_dismean']
    label = pd.merge(label,lucdis_stat,on='user_id',how='left')

    '商家被领取优惠券次数，种类'
    lmc_cnt_set = pd.pivot_table(label,index='merchant_id',values='coupon_id',aggfunc=[getcount,getset]).reset_index()
    lmc_cnt_set.columns = ['merchant_id','lmc_cnt','lmc_set']
    label = pd.merge(label,lmc_cnt_set,on='merchant_id',how='left')

    '商家被领取距离最大、最小、平均'
    lmc_dist_stat = pd.pivot_table(label,index='merchant_id',values='distance',
                                   aggfunc=[np.max,np.min,np.mean]).reset_index()
    lmc_dist_stat.columns = ['merchant_id','lmc_distmax','lmc_distmin','lmc_distmean']
    label = pd.merge(label,lmc_dist_stat,on='merchant_id',how='left')

    '商家被领取折扣率，左值/右值 最大、最小、平均'
    lmc_disc_stat = pd.pivot_table(label,index='merchant_id',values=['dis_left','dis_right'],
                                   aggfunc=[np.max,np.min,np.mean]).reset_index()
    lmc_disc_stat.columns = ['merchant_id','lmc_disc_leftmax','lmc_disc_rightmax','lmc_disc_leftmin',
                             'lmc_disc_rightmin','lmc_disc_leftmean','lmc_disc_rightmean']
    label = pd.merge(label,lmc_disc_stat,on='merchant_id',how='left')

    '商家被领取折扣率最大、最小、平均'
    lmc_dr_stat = pd.pivot_table(label,index='merchant_id',values='dis_rate',
                                 aggfunc=[np.max,np.min,np.mean]).reset_index()
    lmc_dr_stat.columns = ['merchant_id','lmc_drmax','lmc_drmin','lmc_drmean']
    label = pd.merge(label,lmc_dr_stat,on='merchant_id',how='left')

    '优惠券距离最大、最小、平均'
    c_dist_stat = pd.pivot_table(label,index='coupon_id',values='distance',
                                 aggfunc=[np.max,np.min,np.mean]).reset_index()
    c_dist_stat.columns = ['coupon_id','c_distmax','c_distmin','c_distmean']
    label = pd.merge(label,c_dist_stat,on='coupon_id',how='left')

    return label
    pass