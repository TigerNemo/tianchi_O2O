# 天池新人实战赛o2o优惠券使用预测
## 一、数据集划分
使用时间窗口滑动法划分数据集
|     | 特征区间  | 预测区间  |
|  ----  | ----  |   ----   |
| 训练集  | 20160401~20160530 | 20160601~20160630  |
| 验证集  | 201606301~20160430|  20160501~20160530 |
| 测试集  | 20160501~20160630 |  20160701~20160731 |

## 二、特征工程
构建用户、商家、优惠券特征群，以及用户-商家，用户-优惠券，商家-优惠券 三个交叉特征群。
主要包括以下特征：
1.统计特征（最大/最小/平均值/比率 等）
2.排序特征（各个实体对距离，折扣率等的排序）
3.时间特征（日期，时间差等）
  从用户画像的角度来看，统计特征和组合特征，主要分别刻画了用户，商家，优惠券的行为，比如，用户领券次数，商家的热度，优惠券的流行度等等。但是，排序特征，更多地从时间角度，和用户心理角度去考虑。比如说，距离领券时间越近，消费的欲望越强，因为如果领取了优惠券而迟迟没有消费，可能用户本身也忘记了这张优惠券的存在。同时，还有对距离的排序，线下商家与用户的距离越近，肯定要比远的商家消费的概率要大的。

## 三、训练模型
主要使用xgboost模型。该模型精度较高，但训练时间较长。
