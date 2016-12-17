# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-12-15'; 'last updated date: 2016-12-15'
    Email:   '383287471@qq.com'
    Describe: 处理表 9 的  信息 电价方差和电量方差
"""
from __future__ import print_function
from data_util_func import *


def count_elec_std(x):
    """计算电价 方差 和电量方差
        （1）电量方差：
            1）按 月 计算方差
            2）电量为 0 的记录去除掉
    :param x:
    :return:
    """
    # quit()
    # region 1 计算电量 方差
    elec_pq_std = np.nan_to_num(np.std([item for item in x.groupby('RCVBL_YM')['T_PQ'].sum() if item > 0]))
    # endregion
    # region 2 计算电价的 方差
    # 计算每个月 的 电价： 电费/电量
    elec_money_per_degree_a_month = np.nan_to_num(
        (x.groupby('RCVBL_YM')['RCVBL_AMT'].sum() / x.groupby('RCVBL_YM')['T_PQ'].sum()).as_matrix())
    elec_money_per_degree_a_month_std = np.nan_to_num(np.std([item for item in elec_money_per_degree_a_month if item != 0]))

    # endregion

    return '%f,%f' % (elec_pq_std, elec_money_per_degree_a_month_std)


def main(dataset_type='train'):
    if dataset_type == 'train':
        train_data09_merge_label_df = load_data(
            'train_data09_merge_label_df.csv',
            index_col=0,
            header=0,
            encoding='utf8',
            converters={
                'CONS_NO': unicode,
            }
        )

        # print(train_data09_merge_label_df.head())
        train_data09_cons_elec_std_series = train_data09_merge_label_df[:100].groupby('CONS_NO').apply(
            count_elec_std)
        print(train_data09_cons_elec_std_series.head())
        # quit()
        save_data(
            train_data09_cons_elec_std_series,
            'train_data09_cons_elec_std_series.csv',
            index=True,
        )
    elif dataset_type == 'test':
        test_data09_merge_label_df = load_data(
            'test_data09_merge_label_df.csv',
            index_col=0,
            header=0,
            encoding='utf8',
            converters={
                'CONS_NO': unicode,
            }
        )

        # print(test_data09_merge_label_df.head())
        test_data09_cons_elec_std_series = test_data09_merge_label_df[:100].groupby('CONS_NO').apply(
            count_elec_std)
        print(test_data09_cons_elec_std_series.head())
        # quit()
        save_data(
            test_data09_cons_elec_std_series,
            'test_data09_cons_elec_std_series.csv',
            index=True,
        )


    else:
        raise NotImplementedError


if __name__ == '__main__':
    # main(type='train')

    main(dataset_type='train')
