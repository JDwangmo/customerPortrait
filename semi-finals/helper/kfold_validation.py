# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-12-09'; 'last updated date: 2016-12-09'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
from sklearn.cross_validation import KFold
from data_util_func import *
import numpy as np
import pandas as pd
import time
import io

# 数值越大，输出内容越详细
VERBOSE = 2
# [1,2,3]
PLAN_TYPE = 3
# ['rf','lr']
MODEL_NAME = 'rf'

# 执行什么操作： 1 - 验证和测试，2- 验证 、3 - 测试
DO_WHAT = 1

# 验证类型 ： 1- 7/3， 2 - k-fold
VALIDATION_TYPE = 2
N_FOLDS = 5
CLASSIFY_BY_GROUPS = False
GROUP_NAME_LIST = [
    'URBAN_RURAL_FLAG',
    'ELEC_TYPE',
    'LAST_MONTH_PAY_MODE_4bit',
    'IS_CONNECT_TO_08TABLE',
]
GROUP_NAME = GROUP_NAME_LIST[0]
if not CLASSIFY_BY_GROUPS:
    GROUP_NAME = 'ALL'
# 设置正例放大几倍
# auto-自动
# config - 自定义
POSITIVE_EXAMPLE_TIMES_TYPE = 'auto'
# POSITIVE_EXAMPLE_TIMES_TYPE = 'config'
POSITIVE_EXAMPLE_TIMES = 1


def kfold_validation(train_X, train_y, model_name='rf', n_folds=5):
    """ K-fold  验证

    :param train_X:
    :param train_y:
    :param model_name:
    :param n_folds:
    :return:
    """

    train_X_rand = np.random.RandomState(1).permutation(train_X)
    train_y_rand = np.random.RandomState(1).permutation(train_y)
    TP_list, FP_list, TN_list, FN_list = [], [], [], []
    kf = KFold(train_X_rand.shape[0], n_folds=n_folds, random_state=1)
    y_predict_total = []
    # 返回train和test的索引
    for index, (train_index, test_index) in enumerate(kf):
        start = time.time()
        if POSITIVE_EXAMPLE_TIMES_TYPE == 'auto':
            times = sum(train_y_rand == 0) / sum(train_y_rand == 1)
        elif POSITIVE_EXAMPLE_TIMES_TYPE == 'config':
            times = POSITIVE_EXAMPLE_TIMES
        else:
            raise NotImplementedError
        if VERBOSE > 1:
            print('-' * 80)
            print('训练：%d/测试：%d,共:%d,正例*%d倍' % (len(train_index), len(test_index), len(train_X), times))
        train_X_dev_3, train_y_dev_3 = extend_train_data(train_X_rand[train_index], train_y_rand[train_index], n=times)
        rf_model = model_train(
            model_name=model_name,
            train_X=train_X_dev_3,
            train_y=train_y_dev_3.flatten()
        )
        if VERBOSE > 2:
            print(rf_model)

        y_predict, TP, FP, TN, FN = model_predict(rf_model, train_X_rand[test_index], train_y_rand[test_index],
                                                  verbose=VERBOSE)

        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)

        end = time.time()
        if VERBOSE > 1:
            print('validation %d time:%ds' % (index, end - start))

        y_predict_total.append(y_predict)
    # 因为数据被打乱类，现在恢复输入时的顺序
    result = pd.DataFrame(
        data={'TAG1': np.concatenate(y_predict_total)},
        index=np.random.RandomState(1).permutation(len(train_y))
    ).sort_index().as_matrix().flatten()

    if VERBOSE > 0:
        print('-' * 40)
        get_metrics(train_y, result)

    return result


def data_7_3_split(train_X, train_y):
    """ 7/3 验证

    :param train_X:
    :param train_y:
    :return:
    """
    train_X = np.random.RandomState(1).permutation(train_X)
    train_y = np.random.RandomState(1).permutation(train_y)
    # train_is_connect_to_09table = np.random.RandomState(1).permutation(train_is_connect_to_09table)
    index = int(0.7 * len(train_y))

    train_X_dev = train_X[:index]
    train_y_dev = train_y[:index]
    train_X_val = train_X[index:]
    train_y_val = train_y[index:]
    # train_is_connect_to_09table_val = train_is_connect_to_09table[index:]

    # 扩大正例数据 3/5倍，使得数据平衡
    times = sum(train_y == 0) / sum(train_y == 1)
    print(times)
    train_X_dev_3, train_y_dev_3 = extend_train_data(train_X_dev, train_y_dev, n=times)
    print(train_X_dev_3.shape, train_y_dev_3.shape)

    return train_X_dev_3, train_y_dev_3, train_X_val, train_y_val


def validation(train_X, train_y, model_name='rf', type=1):
    """

    :param train_X:
    :param train_y:
    :param model_name:
    :param type: int
        1 - 7/3验证
        2 - kfold验证
    :return:
    """

    # 2 打乱数据
    if type == 1:
        #     7/3验证

        train_X_dev_3, train_y_dev_3, train_X_val, train_y_val = data_7_3_split(train_X, train_y)

        rf_model = model_train(
            model_name=model_name,
            train_X=train_X_dev_3,
            train_y=list(train_y_dev_3)
        )
        print(rf_model)

        model_predict(rf_model, train_X_val, train_y_val)

    elif type == 2:
        return kfold_validation(train_X, train_y, model_name='rf', n_folds=N_FOLDS)

    raise NotImplementedError


def predict(train_X, train_y, test_X, model_name='rf'):
    # 扩大正例数据 3/5倍，使得数据平衡
    if POSITIVE_EXAMPLE_TIMES_TYPE == 'auto':
        times = sum(train_y == 0) / sum(train_y == 1)
    elif POSITIVE_EXAMPLE_TIMES_TYPE == 'config':
        times = POSITIVE_EXAMPLE_TIMES
    else:
        raise NotImplementedError

    print(times)

    train_X_3, train_y_3 = extend_train_data(train_X, train_y, n=times)

    rf_model_total = model_train(
        model_name=model_name,
        train_X=train_X_3,
        train_y=list(train_y_3)
    )
    # pickle.dump(rf_model_total, open('rf_model_total-56features-20161205-001.pkl', 'w'))
    # rf_model_total = pickle.load(open('rf_model_total-56features-20161205-001.pkl', 'r'))
    if VERBOSE > 1:
        print(rf_model_total)

    y_predict_total, TP, FP, TN, FN = model_predict(rf_model_total, test_X, [1] * len(test_X))
    # print(sum(y_predict_total == 1), len(y_predict_total == 1))
    return y_predict_total


def get_group_index(train_data_toclassify, test_data_toclassify=None, type='ELEC_TYPE'):
    if type == 'ELEC_TYPE':
        print('以用电类型分组多分类器训练和预测')
        # 共8组
        groups = [
            [100.0],
            [200.0],
            [201.0],
            [202.0],
            [402.0],
            [403.0],
            [405.0],
        ]
    elif type == 'URBAN_RURAL_FLAG':
        print('以城乡分组多分类器训练和预测')
        # 共2组
        groups = [
            [1],
            [2, 3],
        ]
    elif type == 'LAST_MONTH_PAY_MODE_4bit':
        print('以最后一个缴费方式（4bit）分组多分类器训练和预测')
        # 共3组
        groups = [
            ['0101'],
            ['0202'],
            ['0203'],
        ]
    elif type == 'IS_CONNECT_TO_08TABLE':
        print('以 IS_CONNECT_TO_08TABLE 分组多分类器训练和预测')
        # 共3组
        groups = [
            [0],
            [1],
        ]
    else:
        raise NotImplementedError

    if test_data_toclassify is not None:

        train_data_index = np.arange(train_data_toclassify.shape[0])
        test_data_index = np.arange(test_data_toclassify.shape[0])
        for group in groups:
            print('-' * 50)
            print(group)
            yield (train_data_index[train_data_toclassify[type].apply(lambda x: x in group).as_matrix()],
                   test_data_index[test_data_toclassify[type].apply(lambda x: x in group).as_matrix()]
                   )
    else:
        train_data_index = np.arange(train_data_toclassify.shape[0])
        for group in groups:
            print('-' * 50)
            print(group)
            yield train_data_index[train_data_toclassify[type].apply(lambda x: x in group).as_matrix()]


def model_validation(train_data_toclassify, classify_by_groups=False, group_name='ELEC_TYPE'):
    """ 模型验证

    :param train_data_toclassify:
    :param classify_by_groups: bool
        是否 分组预测
    :return:
    """
    # region 1 数据编码和转为特征矩阵
    train_data_features = data_feature_encoder.fit_transform(
        train_data_toclassify,
    )
    print('-' * 80)
    print(train_data_features.shape)
    train_X = train_data_features.as_matrix()
    train_y = train_data_toclassify['TAG'].as_matrix().flatten()
    print(train_X.shape, train_y.shape)
    # endregion

    start = time.time()
    print('-' * 80)
    # region 2 模型验证 - 总体预测（综合）
    print('综合结果...')
    y_predict_total = validation(train_X, train_y, model_name=MODEL_NAME, type=VALIDATION_TYPE)
    # get_metrics(train_y, y_predict_total)
    # endregion

    # region 3 模型验证 - 分组验证(修正综合结果)
    # y_predict_total = np.asarray(y_predict_total)
    if classify_by_groups:
        for group_index in get_group_index(train_data_toclassify, type=group_name):
            # 特征矩阵
            y_predict = validation(train_X[group_index], train_y[group_index], model_name=MODEL_NAME,
                                   type=VALIDATION_TYPE)
            y_predict_total[group_index] = y_predict

    # endregion
    # region 4 设置验证的预测结果
    train_data_toclassify.loc[:, 'TAG1'] = y_predict_total
    # print(sum(train_data_toclassify['TAG1'] == 1))

    # save_data(train_data_toclassify, 'train_data_toclassify_after_predict.csv')
    temp = train_data_toclassify[train_data_toclassify['TAG1'] == 1]
    save_data(
        temp['CUST_NO'],
        'rf_val_result_%s.csv' % GROUP_NAME,
        header=None
    )

    print('-' * 120)
    get_metrics(train_y, y_predict_total)
    end = time.time()
    print('total validation time:%ds' % (end - start))
    # endregion


def model_test(train_data_toclassify, test_data_toclassify, test_data_sensitive, classify_by_groups=False,
               group_name='ELEC_TYPE'):
    """ 模型预测

    :param train_data_toclassify:
    :param test_data_toclassify:
    :param test_data_sensitive:
    :param classify_by_groups:
    :param group_name:
    :return:
    """
    # region 1 数据编码和转为特征矩阵
    print('-' * 80)
    train_data_features = data_feature_encoder.fit_transform(
        train_data_toclassify,
    )
    train_X = train_data_features.as_matrix()
    train_y = train_data_toclassify['TAG'].as_matrix().flatten()
    print(train_X.shape, train_y.shape)
    test_data_features = data_feature_encoder.transform(
        test_data_toclassify,
    )
    test_data_features.head()
    test_X = test_data_features.as_matrix()
    print(test_X.shape)
    # endregion

    # region 2 模型预测 - 总体预测（综合）
    print('-' * 80)
    start = time.time()
    y_predict_total = predict(train_X, train_y, test_X, model_name=MODEL_NAME)
    end = time.time()
    print('total test time:%ds' % (end - start))

    # endregion

    # region 3 模型预测 - 分组预测(修正综合结果)
    # y_predict_total = np.asarray(y_predict_total)
    if classify_by_groups:
        for group_index in get_group_index(train_data_toclassify, test_data_toclassify, type=group_name):
            # 特征矩阵
            y_predict = predict(train_X[group_index[0]], train_y[group_index[0]], test_X[group_index[1]],
                                model_name=MODEL_NAME)
            y_predict_total[group_index[1]] = y_predict

    print('-' * 120)
    get_metrics([1] * len(test_X), y_predict_total)
    # endregion

    # region 5 保存结果
    print(test_data_sensitive.shape)
    test_data_toclassify.loc[:, 'TAG'] = y_predict_total
    temp = test_data_toclassify[test_data_toclassify['TAG'] == 1]
    # 使用09表 是否连接上 作为规则 过滤用户
    # temp = temp[temp['IS_CONNECT_TO_09TABLE']==1]
    temp = pd.concat([temp, test_data_sensitive])
    print(temp.shape)
    save_data(
        temp['CUST_NO'],
        'rf_result_%s.csv' % GROUP_NAME,
        header=None
    )
    # endregion


def count_accept_content_type_metrics_after_predict():
    """ 统计 分类 预测结果  在小工单类型上 的情况：TP, FP, TN, FN

    :return:
    """
    train_data_toclassify = load_data('train_data_toclassify_after_predict.csv', encoding='utf8')
    print(sum(train_data_toclassify['TAG1'] == 1))

    temp = train_data_toclassify.groupby('ACCEPT_CONTENT_TYPE').apply(
        lambda x: get_metrics2(x['TAG'], x['TAG1'])
    )
    print(temp)
    save_data(
        temp,
        'train&test_train_data_toclassify_accept_content_type_count_df.csv',
        index=True
    )


def merge_multi_result(type=1):
    result_list = [
        # '/home/jdwang/PycharmProjects/customerPortrait/semi-finals/结果/20161213/20161213-12_val_resul_ALL.csv',
        '/home/jdwang/PycharmProjects/customerPortrait/semi-finals/结果/20161214/20161214-1_val_resul.csv',
        # '/home/jdwang/PycharmProjects/customerPortrait/semi-finals/结果/20161213/20161213-8_val_resul.csv',
    ]

    result = []
    for file_path in result_list:
        result += [item.strip() for item in
                   io.open(file_path, encoding='utf8')]

    if type == 1:
        # 1票敏感
        result = list(set(result))
    elif type == 2:
        # 多数敏感
        dd = Counter(result)
        result = map(lambda x: x[0], filter(lambda x: x[1] > 1, dd.items()))
    else:
        raise NotImplementedError
    print(len(result))
    result_dict = {key: 1 for key in result}
    data_to_classify_true_tag = load_data('data_to_classify_true_tag.csv', encoding='utf8',
                                          converters={'CUST_NO': unicode})
    data_to_classify_true_tag['TAG_predict'] = data_to_classify_true_tag['CUST_NO'].map(result_dict)
    data_to_classify_true_tag['TAG_predict'] = data_to_classify_true_tag['TAG_predict'].fillna(0)
    data_to_classify_true_tag['IS_CORRECT'] = data_to_classify_true_tag['TAG_predict'] == data_to_classify_true_tag[
        'TAG']
    save_data(
        data_to_classify_true_tag,
        'result_compare.csv',
    )
    # print(data_to_classify_true_tag.head())
    get_metrics(data_to_classify_true_tag['TAG'].as_matrix(), data_to_classify_true_tag['TAG_predict'].as_matrix(),
                verbose=2)


def add_predict_tag(type=1):
    data_features = load_data('test_features_20161215.csv', encoding='utf8',
                              converters={
                                  'CUST_NO': unicode,
                                  'LAST_MONTH_PAY_MODE': unicode,
                                  'LAST_MONTH_PAY_MODE_4bit': unicode,
                                  'ELEC_DEGREE': unicode,
                              })
    train_data_toclassify_modify_elec_type = load_data('test_RCVBL_AMT&T_PQ.csv',
                                                       encoding='utf8',
                                                       converters={
                                                           'CUST_NO': unicode,
                                                           'LAST_MONTH_PAY_MODE': unicode,
                                                           'LAST_MONTH_PAY_MODE_4bit': unicode,
                                                       }
                                                       )
    # print(data_features.head())
    # print(sum(data_features['ELEC_TYPE'].isnull()))
    # train_sensitiveWorkCount = load_data(
    #     'test_sensitiveWorkCount.csv',
    #     encoding='utf8', converters={'CUST_NO': unicode}
    # )
    train_data_toclassify_modify_elec_type['CUST_NO'] = data_features['CUST_NO'].as_matrix()
    # data_features['ELEC_TYPE'] = train_data_toclassify_modify_elec_type['ELEC_TYPE'].as_matrix()
    save_data(
        train_data_toclassify_modify_elec_type,
        'test_RCVBL_AMT&T_PQ-1.csv'
    )
    # print(sum(data_features['ELEC_TYPE'].isnull()))


    print(data_features.head())
    quit()
    result11_list = [
        '/home/jdwang/PycharmProjects/customerPortrait/semi-finals/结果/20161215/20161215-1.csv',
        # '/home/jdwang/PycharmProjects/customerPortrait/semi-finals/结果/20161214/20161214-1_val_resul.csv',
        # '/home/jdwang/PycharmProjects/customerPortrait/semi-finals/结果/20161213/20161213-8_val_resul.csv',
    ]

    result = []
    for file_path in result_list:
        result += [item.strip() for item in
                   io.open(file_path, encoding='utf8')]

    if type == 1:
        # 1票敏感
        result = list(set(result))
    elif type == 2:
        # 多数敏感
        dd = Counter(result)
        result = map(lambda x: x[0], filter(lambda x: x[1] > 1, dd.items()))
    else:
        raise NotImplementedError

    result_dict = {key: 1 for key in result}
    data_features['TAG_predict'] = data_features['CUST_NO'].map(result_dict)
    data_features['TAG_predict'] = data_features['TAG_predict'].fillna(0)
    # data_features['IS_CORRECT'] = data_features['TAG_predict'] == data_features['TAG']
    save_data(
        data_features,
        'test_features_20161215-TAG.csv',
    )


def main():
    # region 1 恢复
    # 训练集
    train_data01_a_worker_per_user = load_data('train_data01_a_worker_per_user.csv',
                                               encoding='utf8',
                                               converters={
                                                   'CUST_NO': unicode,
                                                   'LAST_MONTH_PAY_MODE': unicode,
                                                   'LAST_MONTH_PAY_MODE_4bit': unicode,
                                                   'MULTI_ELEC_DEGREE': unicode,
                                               }
                                               )
    # (658374, 48)
    print(train_data01_a_worker_per_user.shape)

    # 4-2-2 测试集
    test_data01_a_worker_per_user = load_data('test_data01_a_worker_per_user.csv',
                                              encoding='utf8',
                                              converters={
                                                  'CUST_NO': unicode,
                                                  'LAST_MONTH_PAY_MODE': unicode,
                                                  'LAST_MONTH_PAY_MODE_4bit': unicode,
                                                  'MULTI_ELEC_DEGREE': unicode,
                                              }
                                              )
    print(test_data01_a_worker_per_user.shape)
    # endregion

    # region 2 准备数据和模型构建 和 特征编码  # 数据细分  - 规则匹配 - 交于分类器处理

    print('-' * 80)
    train_data_toclassify, train_data_sensitive = seperate_data_to_classifier(
        train_data01_a_worker_per_user,
        plan_type=PLAN_TYPE
    )


    # sum(temp['CUST_NO'].isnull())
    print(sum(train_data_toclassify['ELEC_TYPE'].isnull()))
    if DO_WHAT in [1, 2]:
        # 验证
        model_validation(train_data_toclassify, classify_by_groups=CLASSIFY_BY_GROUPS, group_name=GROUP_NAME)

    if DO_WHAT in [1, 3]:
        # 测试
        print('-' * 80)
        test_data_toclassify, test_data_sensitive = seperate_data_to_classifier(
            test_data01_a_worker_per_user,
            type='test',
            plan_type=PLAN_TYPE
        )
        test_data_toclassify_modify_elec_type = load_data('test_features-20161213_addElEC_TYPE_by01&09.csv',
                                                          encoding='utf8',
                                                          converters={
                                                              'CUST_NO': unicode,
                                                              'LAST_MONTH_PAY_MODE': unicode,
                                                              'LAST_MONTH_PAY_MODE_4bit': unicode,
                                                          }
                                                          )
        test_data_toclassify['ELEC_TYPE'] = test_data_toclassify_modify_elec_type['ELEC_TYPE'].as_matrix()
        print(sum(test_data_toclassify['ELEC_TYPE'].isnull()))

        model_test(train_data_toclassify, test_data_toclassify, test_data_sensitive,
                   classify_by_groups=CLASSIFY_BY_GROUPS,
                   group_name=GROUP_NAME)

        # endregion


if __name__ == '__main__':
    main()
    # merge_multi_result(1)
    # add_predict_tag()
    # kf = KFold(380542, n_folds=5, random_state=1)
    # # 返回train和test的索引
    # fout = open('validation_index.txt', 'w')
    # val_index_list = []
    # for index, (train_index, test_index) in enumerate(kf):
    #     fout.write(','.join(map(str,train_index))+'\n')
    #     fout.write(','.join(map(str,test_index))+'\n')
    #
    # count_accept_content_type_metrics_after_predict()
