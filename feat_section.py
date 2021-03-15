#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

def getcount(x):
    return x.count()
def getset(x):
    return len(set(x))

def FeatSection(data,label):
    'Feature区间特征'

    '用户领取优惠券次数及种类'
    uc_cnt_set = pd.pivot_table(data,index='user_id',values='coupon_id',aggfunc=[getcount,getset]).reset_index()
    uc_cnt_set.columns = ['user_id','uc_cnt','uc_set']
    label = pd.merge(label,uc_cnt_set,on='user_id',how='left')

    usecp = data[data['date'].isnull() == False]  # 核销优惠券的数据
    dropcp = data[data['date'].isnull() == True]  # 不核销优惠券的数据

    '用户核销优惠券次数及种类'
    uusec_cnt_set = pd.pivot_table(usecp,index='user_id',values='coupon_id',aggfunc=[getcount,getset]).reset_index()
    uusec_cnt_set.columns = ['user_id','uusec_cnt','uusec_set']
    label = pd.merge(label,uusec_cnt_set,on='user_id',how='left')

    '用户领取商家的种类'
    um_set = pd.pivot_table(data,index='user_id',values='merchant_id',aggfunc=getset).reset_index()
    um_set.columns = ['user_id','UM_set']
    label = pd.merge(label,um_set,on='user_id',how='left')

    '用户核销的商家种类'
    uusem_set = pd.pivot_table(usecp,index='user_id',values='merchant_id',aggfunc=getset).reset_index()
    uusem_set.columns = ['user_id','uusem_set']
    label = pd.merge(label,uusem_set,on='user_id',how='left')

    '用户领券距离最大、最小、平均'
    uc_dismax_dismin_dismean = pd.pivot_table(data,index='user_id',values='distance',
                                              aggfunc=[np.max,np.min,np.mean]).reset_index()
    uc_dismax_dismin_dismean.columns = ['user_id','uc_dismax','uc_dismin','uc_dismean']
    label = pd.merge(label,uc_dismax_dismin_dismean,on='user_id',how='left')

    '用户核销距离最大、最小、平均'
    uusec_dismax_dismin_dismean = pd.pivot_table(usecp,index='user_id',values='distance',
                                                 aggfunc=[np.max,np.min,np.mean]).reset_index()
    uusec_dismax_dismin_dismean.columns = ['user_id','uusec_dismax','uusec_dismin','uusec_dismean']
    label = pd.merge(label,uusec_dismax_dismin_dismean,on='user_id',how='left')

    '商家被领取优惠券次数及种类'
    mc_cnt_set = pd.pivot_table(data,index='merchant_id',values='coupon_id',aggfunc=[getcount,getset]).reset_index()
    mc_cnt_set.columns = ['merchant_id','mc_cnt','mc_set']
    label = pd.merge(label,mc_cnt_set,on='merchant_id',how='left')

    '商家被核销优惠券次数及种类'
    musec_cnt_set = pd.pivot_table(usecp,index='merchant_id',values='coupon_id',aggfunc=[getcount,getset]).reset_index()
    musec_cnt_set.columns = ['merchant_id','musec_cnt','musec_set']
    label = pd.merge(label,musec_cnt_set,on='merchant_id',how='left')

    '商家被领取距离最大、最小、平均'
    mc_dismax_dismin_dismean = pd.pivot_table(data,index='merchant_id',values='coupon_id',
                                              aggfunc=[np.max,np.min,np.mean]).reset_index()
    mc_dismax_dismin_dismean.columns = ['merchant','mc_dismax','mc_dismin','mc_dismean']
    label = pd.merge(label,mc_dismax_dismin_dismean,on='merchant_id',how='left')

    '商家被核销距离最大、最小、平均'
    musec_dismax_dismin_dismean = pd.pivot_table(usecp,index='merchant_id',values='coupon_id',
                                                 aggfunc=[np.max,np.min,np.mean]).reset_index()
    musec_dismax_dismin_dismean.columns = ['merchant_id','mc_dismax','mc_dismin','mc_dismean']
    label = pd.merge(label,musec_dismax_dismin_dismean,on='merchant_id',how='left')

    '商家被领取折扣率最大、最小、平均'
    mc_drmax_drmin_drmean = pd.pivot_table(data,index='merchant_id',values='coupon_id',
                                           aggfunc=[np.max,np.min,np.mean]).reset_index()
    mc_drmax_drmin_drmean.columns = ['merchant_id','mc_drmax','mc_drmin','mc_drmean']
    label = pd.merge(label,mc_drmax_drmin_drmean,on='merchant_id',how='left')

    '商家被核销折扣率最大、最小、平均'
    musec_drmax_drmin_drmean = pd.pivot_table(usecp,index='merchant_id',values='coupon_id',
                                              aggfunc=[np.max,np.min,np.mean]).reset_index()
    musec_drmax_drmin_drmean.columns = ['merchant_id','mc_drmax','mc_drmin','mc_drmean']
    label = pd.merge(label,musec_drmax_drmin_drmean,on='merchant_id',how='left')

    '优惠券领取距离最大、最小、平均'
    c_dismax_dismin_dismean = pd.pivot_table(data,index='coupon_id',values='distance',
                                             aggfunc=[np.max,np.min,np.mean]).reset_index()
    c_dismax_dismin_dismean.columns = ['coupon_id','c_dismax','c_dismin','c_dismean']
    label = pd.merge(label,c_dismax_dismin_dismean,on='coupon_id',how='left')

    '优惠券核销距离最大、最小、平均'
    cuse_dismax_dismin_dismean = pd.pivot_table(usecp,index='coupon_id',values='distance',
                                                aggfunc=[np.max,np.min,np.mean]).reset_index()
    cuse_dismax_dismin_dismean.columns = ['coupon_id','cuse_dismax','cuse_dismin','cuse_dismean']
    label = pd.merge(label,cuse_dismax_dismin_dismean,on='coupon_id',how='left')


    return label
    pass

if __name__=='__main__':
    FeatSection()
    pass
