# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-12-09'; 'last updated date: 2016-12-09'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd

N_FOLDS = 5


def kfold_validation(train_X, train_y, n_folds=5):
    train_X_rand = np.random.RandomState(1).permutation(train_X)

    kf = KFold(train_X_rand.shape[0], n_folds=n_folds, random_state=1)
    y_predict_total = []

    for index, (train_index, test_index) in enumerate(kf):
        rf_model = model_train(
            train_X=train_X,
            train_y=train_y.flatten()
        )

        y_predict = model_predict(rf_model, train_X_rand[test_index])

        y_predict_total.append(y_predict)

    result = pd.DataFrame(
        data={'TAG1': np.concatenate(y_predict_total)},
        index=np.random.RandomState(1).permutation(len(train_y))
    ).sort_index().as_matrix().flatten()
    get_metrics(train_y,result)
    return result


def load_data(file_name, header=0, encoding='gbk', converters=None):
    data = pd.read_csv(file_name,
                       sep='\t',
                       encoding=encoding,
                       header=header,
                       quoting=3,
                       converters=converters,
                       )
    return data


def main():
    # 训练集
    train_data_toclassify = load_data('train_features-20161213.csv',
                                      encoding='utf8',
                                      )

    train_X = train_data_toclassify[[
        'NUM_OF_WORKER', 'NUM_OF_SEARCH_ACTION', 'HANDLE_MONTH', 'NUM_OF_IN_SEASON4',
        'AVERAGE_RCVBL_AMT', 'NUM_OF_RCVBL_PENALTY', 'MONEY_PER_DEGREE_STD',
    ]].as_matrix()
    train_y = train_data_toclassify['TAG'].as_matrix().flatten()

    kfold_validation(train_X, train_y, n_folds=N_FOLDS)


def get_metrics(true_y, predict_y, verbose=2):
    from sklearn.metrics import f1_score, precision_score, recall_score

    TP = sum((np.asarray(predict_y) == 1) * (np.asarray(true_y) == 1))
    FP = sum((np.asarray(predict_y) == 1) * (np.asarray(true_y) == 0))
    TN = sum((np.asarray(predict_y) == 0) * (np.asarray(true_y) == 0))
    FN = sum((np.asarray(predict_y) == 0) * (np.asarray(true_y) == 1))
    if verbose > 1:
        print('total:%d' % len(true_y))
        print('TP:%d,FP:%d,TN:%d,FN:%d' % (TP, FP, TN, FN))
        print('f1_score:%f' % f1_score(true_y, predict_y))
        print('precision_score:%f' % precision_score(true_y, predict_y))
        print('recall_score:%f' % recall_score(true_y, predict_y))
        print('accu:%f' % np.mean(true_y == predict_y))

        print('预测为敏感：%d，不敏感：%d' % (sum(predict_y == 1), sum(predict_y == 0)))

    return TP, FP, TN, FN


def model_train(
        train_X=None,
        train_y=None
):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, n_jobs=10, random_state=0)

    model.fit(train_X, train_y)

    return model


def model_predict(model, test_X):
    y_predict = model.predict(test_X)
    return y_predict


if __name__ == '__main__':
    main()
