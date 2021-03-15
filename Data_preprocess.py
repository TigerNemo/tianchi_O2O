#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt

def mk_label():

    ccf_offline_stage1_train = pd.read_csv('ccf_offline_stage1_train.csv')
    # # 筛选出领取优惠券的用户
    ccf_offline_stage1_train = ccf_offline_stage1_train[
         ccf_offline_stage1_train['Date_received'].isnull().values == False]

    ## 筛选出是否使用优惠券消费的用户，即正负样本
    # 用户有领取优惠券但没有核销，负样本，标0
    part1 = ccf_offline_stage1_train[
        ccf_offline_stage1_train['Date'].isnull().values == True]
    part1['label'] = 0
    part1['Date_received'] = part1['Date_received'].map(lambda x: str(int(x)))
    part1['Date_received'] = pd.to_datetime(part1['Date_received'],format='%Y-%m-%d')
    part1['Date'] = pd.to_datetime(part1['Date'],format='%Y-%m-%d')

    # # 用户在领取优惠券后15天内使用则为正样本（标1），否则为负样本
    part2 = ccf_offline_stage1_train[ccf_offline_stage1_train['Date'].isnull().values == False]
    part2['Date_received'] = part2['Date_received'].map(lambda x: str(int(x)))
    part2['Date_received'] = pd.to_datetime(part2['Date_received'], format='%Y-%m-%d')
    part2['Date'] = part2['Date'].map(lambda x: str(int(x)))
    part2['Date'] = pd.to_datetime(part2['Date'],format='%Y-%m-%d')
                   # 转换为时间类型进行，对天数进行相加减

    part2['label'] = [0 if int(i.days) > 15 else 1 for i in (part2['Date']-part2['Date_received'])]

    ccf_offline_stage1_train = part1.append(part2)
    print(ccf_offline_stage1_train)
    ccf_offline_stage1_train.to_csv('row_train.csv',index=False)
    pass
def split():
    row_train = pd.read_csv('row_train.csv')
    '数据预处理和数据集划分'
    row_train['Date_received'] = [str(i)[:10].replace('-','') for i in row_train['Date_received']]
    row_train['Date_received'] = row_train['Date_received'].map(lambda x:int(x))

    row_train['Distance'] = row_train['Distance'].fillna(-1)  # 缺失值填充
    row_train.rename(columns={'User_id' : 'user_id',
                              'Merchant_id' : 'merchant_id',
                              'Coupon_id' : 'coupon_id',
                              'Discount_rate' : 'discount_rate',
                              'Distance' : 'distance',
                              'Date_received' : 'date_received',
                              'Date' : 'date'},inplace=True)  # 原地替换，节省内存
    # print(row_train['date'])
    def getrate(x):
        if len(x)==2:
            x[0] = int(x[0])
            x[1] = int(x[1])
            tmp = (x[0] - x[1])/x[0]
            return tmp
        pass

    row_train['discount'] = row_train['discount_rate'].map(lambda x: x.split(':'))  # 筛选出满减类型
    row_train['dis_left'] = row_train['discount'].map(lambda x: int(x[0]) if len(x) == 2 else -1)
    row_train['dis_right'] = row_train['discount'].map(lambda x: int(x[1]) if len(x) == 2 else -1)
    row_train['dis_rate'] = row_train['discount'].map(lambda x: getrate(x))
    del row_train['discount']

    '按领券日期划分数据集'

    '训练集'
    train_feat = row_train[(row_train['date_received'] >= 20160401) & (row_train['date_received'] <= 20160530)]
    train_feat.to_csv('train_feat.csv',index=False)
    train_label = row_train[(row_train['date_received'] >= 20160601) & (row_train['date_received'] <= 20160630)]
    train_label.to_csv('train_label.csv',index=False)

    '验证集'
    val_feat = row_train[(row_train['date_received'] >= 20160301) & (row_train['date_received'] <= 20160430)]
    val_feat.to_csv('val_feat.csv',index=False)
    val_label = row_train[(row_train['date_received'] >= 20160501) & (row_train['date_received'] <= 20160530)]
    val_label.to_csv('val_label.csv',index=False)

    '测试集'
    test_feat = row_train[(row_train['date_received'] >= 20160501) & (row_train['date_received'] <= 20160630)]
    test_feat.to_csv('test_feat.csv',index=False)

    '线上提交区间数据处理'
    testlabel = pd.read_csv('ccf_offline_stage1_test_revised.csv')
    testlabel['Distance'] = testlabel['Distance'].fillna(-1)

    testlabel.rename(columns={'User_id' : 'user_id',
                              'Merchant_id' : 'merchant_id',
                              'Coupon_id' : 'coupon_id',
                              'Discount_rate' : 'discount_rate',
                              'Distance' : 'distance',
                              'Date_received' : 'date_received'},inplace=True)
    testlabel['discount'] = testlabel['discount_rate'].map(lambda x: x.split(':'))
    testlabel['dis_left'] = testlabel['discount'].map(lambda x: int(x[0]) if len(x) == 2 else -1)
    testlabel['dis_right'] = testlabel['discount'].map(lambda x: int(x[1]) if len(x) ==2 else -1)
    testlabel['dis_rate'] = testlabel['discount'].map(lambda x: getrate(x))
    del testlabel['discount']  # 实现满减类型的分离转化，并删除原来的discount

    testlabel.to_csv('test_label.csv',index=False)

    pass

def getcount(x):
    return x.count()
def getset(x):
    return len(set(x))

def FeatSection(data,label):
    'Feature区间特征'

    '1.用户领取优惠券次数及种类'
    uc_cnt_set = pd.pivot_table(data,index='user_id',values='coupon_id',aggfunc=[getcount,getset]).reset_index()
    uc_cnt_set.columns = ['user_id','uc_cnt','uc_set']
    label = pd.merge(label,uc_cnt_set,on='user_id',how='left')

    usecp = data[data['date'].isnull() == False]  # 核销优惠券的数据
    dropcp = data[data['date'].isnull() == True]  # 不核销优惠券的数据

    '2.用户核销优惠券次数及种类'
    uusec_cnt_set = pd.pivot_table(usecp,index='user_id',values='coupon_id',aggfunc=[getcount,getset]).reset_index()
    uusec_cnt_set.columns = ['user_id','uusec_cnt','uusec_set']
    label = pd.merge(label,uusec_cnt_set,on='user_id',how='left')

    '3.用户领取商家的种类'
    um_set = pd.pivot_table(data,index='user_id',values='merchant_id',aggfunc=getset).reset_index()
    um_set.columns = ['user_id','UM_set']
    label = pd.merge(label,um_set,on='user_id',how='left')

    '4.用户核销的商家种类'
    uusem_set = pd.pivot_table(usecp,index='user_id',values='merchant_id',aggfunc=getset).reset_index()
    uusem_set.columns = ['user_id','uusem_set']
    label = pd.merge(label,uusem_set,on='user_id',how='left')

    '5.用户领券距离最大、最小、平均'
    uc_dismax_dismin_dismean = pd.pivot_table(data,index='user_id',values='distance',
                                              aggfunc=[np.max,np.min,np.mean]).reset_index()
    uc_dismax_dismin_dismean.columns = ['user_id','uc_dismax','uc_dismin','uc_dismean']
    label = pd.merge(label,uc_dismax_dismin_dismean,on='user_id',how='left')

    '6.用户核销距离最大、最小、平均'
    uusec_dismax_dismin_dismean = pd.pivot_table(usecp,index='user_id',values='distance',
                                                 aggfunc=[np.max,np.min,np.mean]).reset_index()
    uusec_dismax_dismin_dismean.columns = ['user_id','uusec_dismax','uusec_dismin','uusec_dismean']
    label = pd.merge(label,uusec_dismax_dismin_dismean,on='user_id',how='left')

    '7.商家被领取优惠券次数及种类'
    mc_cnt_set = pd.pivot_table(data,index='merchant_id',values='coupon_id',aggfunc=[getcount,getset]).reset_index()
    mc_cnt_set.columns = ['merchant_id','mc_cnt','mc_set']
    label = pd.merge(label,mc_cnt_set,on='merchant_id',how='left')

    '8.商家被核销优惠券次数及种类'
    musec_cnt_set = pd.pivot_table(usecp,index='merchant_id',values='coupon_id',aggfunc=[getcount,getset]).reset_index()
    musec_cnt_set.columns = ['merchant_id','musec_cnt','musec_set']
    label = pd.merge(label,musec_cnt_set,on='merchant_id',how='left')

    '9.商家被领取距离最大、最小、平均'
    mc_dismax_dismin_dismean = pd.pivot_table(data,index='merchant_id',values='coupon_id',
                                              aggfunc=[np.max,np.min,np.mean]).reset_index()
    mc_dismax_dismin_dismean.columns = ['merchant_id','mc_dismax','mc_dismin','mc_dismean']
    label = pd.merge(label,mc_dismax_dismin_dismean,on='merchant_id',how='left')

    '10.商家被核销距离最大、最小、平均'
    musec_dismax_dismin_dismean = pd.pivot_table(usecp,index='merchant_id',values='coupon_id',
                                                 aggfunc=[np.max,np.min,np.mean]).reset_index()
    musec_dismax_dismin_dismean.columns = ['merchant_id','musec_dismax','musec_dismin','musec_dismean']
    label = pd.merge(label,musec_dismax_dismin_dismean,on='merchant_id',how='left')

    '11.商家被领取折扣率最大、最小、平均'
    mc_drmax_drmin_drmean = pd.pivot_table(data,index='merchant_id',values='coupon_id',
                                           aggfunc=[np.max,np.min,np.mean]).reset_index()
    mc_drmax_drmin_drmean.columns = ['merchant_id','mc_drmax','mc_drmin','mc_drmean']
    label = pd.merge(label,mc_drmax_drmin_drmean,on='merchant_id',how='left')

    '12.商家被核销折扣率最大、最小、平均'
    musec_drmax_drmin_drmean = pd.pivot_table(usecp,index='merchant_id',values='coupon_id',
                                              aggfunc=[np.max,np.min,np.mean]).reset_index()
    musec_drmax_drmin_drmean.columns = ['merchant_id','musec_drmax','musec_drmin','musec_drmean']
    label = pd.merge(label,musec_drmax_drmin_drmean,on='merchant_id',how='left')

    '13.优惠券领取距离最大、最小、平均'
    c_dismax_dismin_dismean = pd.pivot_table(data,index='coupon_id',values='distance',
                                             aggfunc=[np.max,np.min,np.mean]).reset_index()
    c_dismax_dismin_dismean.columns = ['coupon_id','c_dismax','c_dismin','c_dismean']
    label = pd.merge(label,c_dismax_dismin_dismean,on='coupon_id',how='left')

    '14.优惠券核销距离最大、最小、平均'
    cuse_dismax_dismin_dismean = pd.pivot_table(usecp,index='coupon_id',values='distance',
                                                aggfunc=[np.max,np.min,np.mean]).reset_index()
    cuse_dismax_dismin_dismean.columns = ['coupon_id','cuse_dismax','cuse_dismin','cuse_dismean']
    label = pd.merge(label,cuse_dismax_dismin_dismean,on='coupon_id',how='left')


    return label
    pass

def LabelSection(label):
    'label区间特征'

    '1.用户领取优惠券的次数，种类'
    luc_cnt_set = pd.pivot_table(label,index='user_id',values='coupon_id',aggfunc=[getcount,getset]).reset_index()
    luc_cnt_set.columns = ['user_id','luc_cnt','luc_set']
    label = pd.merge(label,luc_cnt_set,on='user_id',how='left')

    '2.用户领取商家的次数，种类'
    lum_set = pd.pivot_table(label,index='user_id',values='merchant_id',aggfunc=[getcount,getset]).reset_index()
    lum_set.columns = ['user_id','lum_cnt','lum_set']
    label = pd.merge(label,lum_set,on='user_id',how='left')

    '3.用户领取距离的最大、最小、平均'
    lucdis_stat = pd.pivot_table(label,index='user_id',values='coupon_id',aggfunc=[np.max,np.min,np.mean]).reset_index()
    lucdis_stat.columns = ['user_id','luc_dismax','luc_dismin','luc_dismean']
    label = pd.merge(label,lucdis_stat,on='user_id',how='left')

    '4.商家被领取优惠券次数，种类'
    lmc_cnt_set = pd.pivot_table(label,index='merchant_id',values='coupon_id',aggfunc=[getcount,getset]).reset_index()
    lmc_cnt_set.columns = ['merchant_id','lmc_cnt','lmc_set']
    label = pd.merge(label,lmc_cnt_set,on='merchant_id',how='left')

    '5.商家被领取距离最大、最小、平均'
    lmc_dist_stat = pd.pivot_table(label,index='merchant_id',values='distance',
                                   aggfunc=[np.max,np.min,np.mean]).reset_index()
    lmc_dist_stat.columns = ['merchant_id','lmc_distmax','lmc_distmin','lmc_distmean']
    label = pd.merge(label,lmc_dist_stat,on='merchant_id',how='left')

    '6.商家被领取折扣率，左值/右值 最大、最小、平均'
    lmc_disc_stat = pd.pivot_table(label,index='merchant_id',values=['dis_left','dis_right'],
                                   aggfunc=[np.max,np.min,np.mean]).reset_index()
    lmc_disc_stat.columns = ['merchant_id','lmc_disc_leftmax','lmc_disc_rightmax','lmc_disc_leftmin',
                             'lmc_disc_rightmin','lmc_disc_leftmean','lmc_disc_rightmean']
    label = pd.merge(label,lmc_disc_stat,on='merchant_id',how='left')

    '7.商家被领取折扣率最大、最小、平均'
    lmc_dr_stat = pd.pivot_table(label,index='merchant_id',values='dis_rate',
                                 aggfunc=[np.max,np.min,np.mean]).reset_index()
    lmc_dr_stat.columns = ['merchant_id','lmc_drmax','lmc_drmin','lmc_drmean']
    label = pd.merge(label,lmc_dr_stat,on='merchant_id',how='left')

    '8.优惠券距离最大、最小、平均'
    c_dist_stat = pd.pivot_table(label,index='coupon_id',values='distance',
                                 aggfunc=[np.max,np.min,np.mean]).reset_index()
    c_dist_stat.columns = ['coupon_id','c_distmax','c_distmin','c_distmean']
    label = pd.merge(label,c_dist_stat,on='coupon_id',how='left')

    return label
    pass

def featgen_train():
    '训练集特征生成'

    feat = pd.read_csv('train_feat.csv')
    label = pd.read_csv('train_label.csv')

    label = FeatSection(data=feat,label=label)
    label = LabelSection(label=label)

    label = label.fillna(-1)

    return label
    pass

def featgen_test():
    '测试集特征生成'

    feat = pd.read_csv('test_feat.csv')
    label = pd.read_csv('test_label.csv')

    label = FeatSection(data=feat,label=label)
    label = LabelSection(label=label)

    label = label.fillna(-1)
    return label
    pass

def featgen_val():
    '验证集特征生成'

    feat = pd.read_csv('val_feat.csv')
    label = pd.read_csv('val_label.csv')

    label = FeatSection(data=feat,label=label)
    label = LabelSection(label=label)

    label = label.fillna(-1)
    return label
    pass

def mulval():
    train = featgen_train()
    val = featgen_val()
    test = featgen_test()

    train.to_csv('Train.csv',index=False)
    val.to_csv('Val.csv',index=False)
    test.to_csv('Test.csv',index=False)
    pass

def XgbTest(train,test):
    train_x = train.drop(['user_id','merchant_id','coupon_id','date_received','discount_rate','date','label'],axis=1)
    train_y = train['label']
    test_id = test[['user_id','coupon_id','date_received']]
    test_x = test.drop(['user_id','merchant_id','coupon_id','date_received','discount_rate',],axis=1)
    # xgb矩阵赋值
    xgb_train = xgb.DMatrix(train_x,label=train_y.values)
    xgb_test = xgb.DMatrix(test_x)
    params = {'booster' : 'gbtree',  # 基于树模型
              'objective' : 'binary:logistic',  # 二分类逻辑回归
              'eta' : '0.01',   # 学习率，通过减少每一步的权重，提高模型的鲁棒性
              'eval_metric' : 'auc',  # 模型的评估指标
              'subsample' : 0.8,   # 控制每个树随机采样的比例
              'colsample_bytree' : 0.8,  # 控制每个树随机采样的例数（特征）的占比
              'scale_pos_weight' : 1,   # 在样本不平衡时>0,使算法更快收敛
              'min_child_weight' : 18,  # 最小样本权重的和，用于避免过拟合
              }
    model = xgb.train(params,xgb_train,num_boost_round=1200)
    result = model.predict(xgb_test)
    test_id['Probability'] = result
    test_id.rename(columns={'user_id' : 'User_id',
                            'coupon_id' : 'Coupon_id',
                            'date_received' : 'Date_received'},inplace=True)

    test_id.to_csv('xgb.csv',index=False)
    plot_importance(model)
    plt.show()
    pass

def FromSave():
    train = pd.read_csv('Train.csv')
    # val = pd.read_csv('Val.csv')
    test = pd.read_csv('Test.csv')
    # print(train['date'])
    XgbTest(train=train,test=test)
    pass

if __name__=='__main__':
    featgen_train()
    featgen_val()
    featgen_test()
    mulval()
    FromSave()
    pass