# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-12-03'; 'last updated date: 2016-12-03'
    Email:   '383287471@qq.com'
    Describe: 特征工程类
"""
from __future__ import print_function

from datetime import datetime
import pandas as pd
import io


class FeatureEncoder(object):
    """ 特征工程类:
        主要 将 类别型 的特征 进行 onehot 编码
             - 每个用户的工单数

    """

    def __init__(self,
                 num_of_worker=False,
                 num_of_search_action=False,
                 max_num_month_search_action=False,
                 num_of_used_pay_mode=False,
                 num_of_in_season4=False,
                 num_of_rcvbl_penalty=False,
                 normal_num_of_rcvbl_penalty=False,

                 num_of_sensitive_workers=False,

                 average_rcvbl_amt=False,
                 normal_average_rcvbl_amt=False,

                 money_per_degree_std=False,
                 # 月电量 方差
                 month_pq_std=False,

                 is_connect_to_06table=False,
                 is_connect_to_07table=False,
                 is_connect_to_08table=False,
                 is_connect_to_09table=False,
                 is_penalty=False,
                 is_exceeding_rcvbl_ym_ge_1mon=False,

                 # 是否包含缴费方式 020311
                 is_pay_mode_contains_020311=False,
                 # 是否包含缴费方式 010101
                 is_pay_mode_contains_010101=False,
                 # 是否包含缴费方式 020261
                 is_pay_mode_contains_020261=False,

                 is_rcvbl_amt_t_pq_equal0=False,
                 is_rcvbl_amt_equal0_t_pq_grater0=False,
                 is_rcvbl_amt_lower0=False,

                 is_hebiao_user_09table=False,
                 is_elec_eq_zero_09table=False,
                 is_seperate_time_09table=False,
                 is_mid_change_09table=False,
                 # 电价最高档数
                 elec_degree_09table=False,

                 handle_year=False,
                 handle_month=False,
                 handle_day=False,
                 handle_hour=False,

                 cust_no_3bit=False,
                 busi_type_code=False,
                 urban_rural_flag=False,
                 city_org_no=False,
                 elec_type=False,

                 accept_content_type=False,
                 multi_accept_content_type=False,
                 accept_content_type_version='0.01',

                 cont_type_06table=False,
                 status_06table=False,

                 cons_status_07table=False,
                 rca_flag_07table=False,

                 org_no_9bit_09table=False,
                 org_no_7bit_09table=False,

                 # 用户最后一个月的缴费方式
                 last_month_pay_mode_09table=False,
                 last_month_pay_mode_4bit_09table=False,

                 # 用户缴费方式 的 转变路线
                 pay_mode_change_clue_09table=False,
                 pay_mode_4bit_change_clue_09table=False,

                 ):
        """
            初始化参数
        """

        self.num_of_worker = num_of_worker
        self.num_of_search_action = num_of_search_action
        self.max_num_month_search_action = max_num_month_search_action
        self.num_of_used_pay_mode = num_of_used_pay_mode
        self.money_per_degree_std = money_per_degree_std
        self.month_pq_std = month_pq_std

        self.num_of_in_season4 = num_of_in_season4
        self.num_of_sensitive_workers = num_of_sensitive_workers
        self.num_of_rcvbl_penalty = num_of_rcvbl_penalty
        self.normal_num_of_rcvbl_penalty = normal_num_of_rcvbl_penalty

        self.average_rcvbl_amt = average_rcvbl_amt
        self.normal_average_rcvbl_amt = normal_average_rcvbl_amt

        self.is_connect_to_06table = is_connect_to_06table
        self.is_connect_to_07table = is_connect_to_07table
        self.is_connect_to_08table = is_connect_to_08table
        self.is_connect_to_09table = is_connect_to_09table

        self.is_penalty = is_penalty
        self.is_exceeding_rcvbl_ym_ge_1mon = is_exceeding_rcvbl_ym_ge_1mon

        self.is_pay_mode_contains_020311 = is_pay_mode_contains_020311
        self.is_pay_mode_contains_010101 = is_pay_mode_contains_010101
        self.is_pay_mode_contains_020261 = is_pay_mode_contains_020261

        self.is_rcvbl_amt_t_pq_equal0 = is_rcvbl_amt_t_pq_equal0
        self.is_rcvbl_amt_equal0_t_pq_grater0 = is_rcvbl_amt_equal0_t_pq_grater0
        self.is_rcvbl_amt_lower0 = is_rcvbl_amt_lower0

        self.is_hebiao_user_09table = is_hebiao_user_09table
        self.is_elec_eq_zero_09table = is_elec_eq_zero_09table
        self.elec_degree_09table = elec_degree_09table
        self.is_seperate_time_09table = is_seperate_time_09table
        self.is_mid_change_09table = is_mid_change_09table

        self.handle_year = handle_year
        self.handle_month = handle_month
        self.handle_day = handle_day
        self.handle_hour = handle_hour

        self.cust_no_3bit = cust_no_3bit
        self.busi_type_code = busi_type_code
        self.urban_rural_flag = urban_rural_flag
        self.city_org_no = city_org_no
        self.elec_type = elec_type
        self.accept_content_type = accept_content_type
        self.multi_accept_content_type = multi_accept_content_type
        self.accept_content_type_version = accept_content_type_version

        self.cont_type_06table = cont_type_06table
        self.status_06table = status_06table
        self.cons_status_07table = cons_status_07table
        self.rca_flag_07table = rca_flag_07table

        self.org_no_7bit_09table = org_no_7bit_09table
        self.org_no_9bit_09table = org_no_9bit_09table
        self.last_month_pay_mode_09table = last_month_pay_mode_09table
        self.last_month_pay_mode_4bit_09table = last_month_pay_mode_4bit_09table

        self.pay_mode_change_clue_09table = pay_mode_change_clue_09table
        self.pay_mode_4bit_change_clue_09table = pay_mode_4bit_change_clue_09table

        if self.normal_num_of_rcvbl_penalty:
            self.num_of_rcvbl_penalty_mean = None
            self.num_of_rcvbl_penalty_std = None

        if self.normal_average_rcvbl_amt:
            self.average_rcvbl_amt_mean = None
            self.average_rcvbl_amt_std = None

        if self.cust_no_3bit:
            self.cust_no_3bit_list = None
        if self.busi_type_code:
            self.busi_type_code_list = None
        if self.urban_rural_flag:
            self.urban_rural_flag_list = None
        if self.city_org_no:
            self.city_org_no_list = None
        if self.elec_type:
            self.elec_type_list = None
        if self.accept_content_type:
            # 小工单类型 特征列表
            self.accept_content_type_list = None
        if self.multi_accept_content_type:
            # 小工单类型 特征列表
            self.multi_accept_content_type_list = None

        if self.cont_type_06table:
            # 小工单类型 特征列表
            self.cont_type_06table_list = None
        if self.status_06table:
            # 小工单类型 特征列表
            self.status_06table_list = None
        if self.cons_status_07table:
            # 小工单类型 特征列表
            self.cons_status_07table_list = None
        if self.rca_flag_07table:
            # 小工单类型 特征列表
            self.rca_flag_07table_list = None
        if self.org_no_7bit_09table:
            # ORG_NO 截断成7位
            self.org_no_7bit_09table_list = None
        if self.org_no_9bit_09table:
            # ORG_NO 截断成9位,出现一个用户多org，则取多数
            self.org_no_9bit_09table_list = None
        if self.last_month_pay_mode_09table:
            # 用户最后一个月的缴费方式
            self.last_month_pay_mode_09table_list = None
        if self.last_month_pay_mode_4bit_09table:
            # 用户最后一个月的缴费方式
            self.last_month_pay_mode_4bit_09table_list = None
        if self.pay_mode_change_clue_09table:
            # 用户缴费方式 的 转变路线
            self.pay_mode_change_clue_09table_list = None
        if self.pay_mode_4bit_change_clue_09table:
            # 用户缴费方式 的 转变路线
            self.pay_mode_4bit_change_clue_09table_list = None

    def fit(self, train_data):
        """拟合数据,取得各个 类别型特征 的 特征值列表

        :param train_data: pd.DataFrame()
        :return:
        """
        if self.num_of_rcvbl_penalty and self.normal_num_of_rcvbl_penalty:
            self.num_of_rcvbl_penalty_mean = train_data['NUM_OF_RCVBL_PENALTY'].mean()
            self.num_of_rcvbl_penalty_std = train_data['NUM_OF_RCVBL_PENALTY'].std()

        if self.average_rcvbl_amt and self.normal_average_rcvbl_amt:
            self.average_rcvbl_amt_mean = train_data['AVERAGE_RCVBL_AMT'].mean()
            self.average_rcvbl_amt_std = train_data['AVERAGE_RCVBL_AMT'].std()

        if self.cust_no_3bit:
            self.cust_no_3bit_list = sorted(
                [item.strip() for item in
                 io.open(
                     'features_selected_to_predict/cust_no_3bit_selected_to_predict.txt', encoding='utf8')])

        if self.busi_type_code:
            # train_data['BUSI_TYPE_CODE'].notnull() 这一步是为了去除空值
            # self.busi_type_code_list = sorted(
            #     list(train_data.loc[train_data['BUSI_TYPE_CODE'].notnull(), 'BUSI_TYPE_CODE'].unique()))
            # 自定义
            self.busi_type_code_list = sorted([15, 5, 18, 3])
        if self.urban_rural_flag:
            self.urban_rural_flag_list = sorted(
                list(train_data.loc[train_data['URBAN_RURAL_FLAG'].notnull(), 'URBAN_RURAL_FLAG'].unique()))

        if self.city_org_no:
            # self.city_org_no_list = sorted(
            #     list(train_data.loc[train_data['CITY_ORG_NO'].notnull(), 'CITY_ORG_NO'].unique()))
            # 自定义
            self.city_org_no_list = sorted(
                [33401, 33405, 33406, 33403, 33408, 33402, 33404, 33407, 33420])
        if self.elec_type:
            # self.elec_type_list = sorted(list(train_data.loc[train_data['ELEC_TYPE'].notnull(), 'ELEC_TYPE'].unique()))
            # 自定义
            self.elec_type_list = sorted(
                [100.0, 200.0, 202.0, 405.0, 402.0, 400.0, 401.0, 203.0, 403.0, 201.0, 300.0, 301.0])

        if self.accept_content_type:
            # 小工单类型 特征列表
            # 从文件读取

            self.accept_content_type_list = sorted(
                [item.strip() for item in
                 io.open(
                     'features_selected_to_predict/accept_content_type_selected_to_predict%s.txt' %
                     self.accept_content_type_version,
                     encoding='utf8')])

        if self.multi_accept_content_type:
            # 小工单类型 特征列表
            # 从文件读取

            self.multi_accept_content_type_list = sorted(
                [item.strip() for item in
                 io.open(
                     'features_selected_to_predict/accept_content_type_selected_to_predict%s.txt' %
                     self.accept_content_type_version,
                     encoding='utf8')])

        if self.cont_type_06table:
            # self.elec_type_list = sorted(list(train_data.loc[train_data['ELEC_TYPE'].notnull(), 'ELEC_TYPE'].unique()))
            # 自定义
            self.cont_type_06table_list = sorted([1])

        if self.status_06table:
            # self.elec_type_list = sorted(list(train_data.loc[train_data['ELEC_TYPE'].notnull(), 'ELEC_TYPE'].unique()))
            # 自定义
            self.status_06table_list = sorted([1, 2])

        if self.cons_status_07table:
            # self.elec_type_list = sorted(list(train_data.loc[train_data['ELEC_TYPE'].notnull(), 'ELEC_TYPE'].unique()))
            # 自定义
            self.cons_status_07table_list = sorted([1, 2])

        if self.rca_flag_07table:
            # self.elec_type_list = sorted(list(train_data.loc[train_data['ELEC_TYPE'].notnull(), 'ELEC_TYPE'].unique()))
            # 自定义
            self.rca_flag_07table_list = sorted([1])
        if self.org_no_7bit_09table:
            self.org_no_7bit_09table_list = sorted(
                list(train_data.loc[train_data['ORG_NO_7bit'].notnull(), 'ORG_NO_7bit'].unique()))

        if self.org_no_9bit_09table:
            self.org_no_9bit_09table_list = sorted(
                list(train_data.loc[train_data['ORG_NO_9bit'].notnull(), 'ORG_NO_9bit'].unique()))

        if self.last_month_pay_mode_09table:
            # 自定义
            self.last_month_pay_mode_09table_list = sorted(['010101', '020261', '020311', '020331', '010106'])

        if self.last_month_pay_mode_4bit_09table:
            # 自定义
            self.last_month_pay_mode_4bit_09table_list = sorted(['010101', '020261', '020311', '020331', '010106'])

        if self.pay_mode_change_clue_09table:
            # 自定义
            self.pay_mode_change_clue_09table_list = sorted([item.strip() for item in io.open(
                'features_selected_to_predict/pay_mode_change_clue_selected_to_predict.txt',
                encoding='utf8')])

            # self.pay_mode_change_clue_09table_list = sorted(
            #     list(train_data.loc[train_data['PAY_MODE_CHANGE_CLUE'].notnull(), 'PAY_MODE_CHANGE_CLUE'].unique()))

        if self.pay_mode_4bit_change_clue_09table:
            # 自定义
            self.pay_mode_4bit_change_clue_09table_list = sorted([item.strip() for item in io.open(
                'features_selected_to_predict/pay_mode_4bit_change_clue_selected_to_predict.txt',
                encoding='utf8')])

        return self

    def transform(self, data):

        data_features = pd.DataFrame()
        # region 直接特征提取
        if self.num_of_worker:
            print('工单数 1')
            data_features = pd.concat([data_features, data['NUM_OF_WORKER']], axis=1)

        if self.num_of_search_action:
            print('查询电费次数 1')
            data_features = pd.concat([data_features, data['NUM_OF_SEARCH_ACTION']], axis=1)

        if self.max_num_month_search_action:
            print('最大 月[查询电费]次数 1')
            data_features = pd.concat([data_features, data['MAX_NUM_MONTH_SEARCH_ACTION']], axis=1)

        if self.num_of_used_pay_mode:
            # 该用户有几种缴费方式
            print('缴费方式 次数 1')
            data_features = pd.concat([data_features, data['MAX_NUM_MONTH_SEARCH_ACTION']], axis=1)

        if self.num_of_in_season4:
            print('第四季度记录数 1')
            data_features = pd.concat([data_features, data['NUM_OF_IN_SEASON4']], axis=1)

        if self.num_of_sensitive_workers:
            print('含敏感词的工单记录数 1')
            data_features = pd.concat([data_features, data['NUM_OF_SENSITIVE_WORKERS']], axis=1)

        if self.num_of_rcvbl_penalty:
            print('应收违约金的次数 1')
            if self.normal_num_of_rcvbl_penalty:
                temp = (data['NUM_OF_RCVBL_PENALTY'] - self.num_of_rcvbl_penalty_mean) / self.num_of_rcvbl_penalty_std
            else:
                temp = data['NUM_OF_RCVBL_PENALTY']

            data_features = pd.concat([data_features,
                                       temp
                                       ], axis=1)

        if self.average_rcvbl_amt:
            print('平均电费 1')
            if self.normal_average_rcvbl_amt:
                temp = (data['AVERAGE_RCVBL_AMT'] - self.average_rcvbl_amt_mean) / self.average_rcvbl_amt_std
            else:
                temp = data['AVERAGE_RCVBL_AMT']

            data_features = pd.concat([data_features,
                                       temp
                                       ], axis=1)

        if self.money_per_degree_std:
            # 该用户有几种缴费方式
            print('月电价方差 1')
            data_features = pd.concat([data_features, data['MONEY_PER_DEGREE_STD']], axis=1)

        if self.month_pq_std:
            # 该用户有几种缴费方式
            print('月电量方差 1')
            data_features = pd.concat([data_features, data['MONTH_PQ_STD']], axis=1)

        if self.is_connect_to_06table:
            print('是否连接上06表 1')
            data_features = pd.concat([data_features, data['IS_CONNECT_TO_06TABLE']], axis=1)

        if self.is_connect_to_07table:
            print('是否连接上07表 1')
            data_features = pd.concat([data_features, data['IS_CONNECT_TO_07TABLE']], axis=1)

        if self.is_connect_to_08table:
            print('是否连接上08表 1')
            data_features = pd.concat([data_features, data['IS_CONNECT_TO_08TABLE']], axis=1)

        if self.is_connect_to_09table:
            print('是否连接上09表 1')
            data_features = pd.concat([data_features, data['IS_CONNECT_TO_09TABLE']], axis=1)

        if self.is_pay_mode_contains_020311:
            print('是否包含缴费方式 020311 1')
            data_features = pd.concat([data_features, data['IS_PAY_MODE_CONTAINS_020311']], axis=1)

        if self.is_pay_mode_contains_010101:
            print('是否包含缴费方式 010101 1')
            data_features = pd.concat([data_features, data['IS_PAY_MODE_CONTAINS_010101']], axis=1)

        if self.is_pay_mode_contains_020261:
            print('是否包含缴费方式 020261 1')
            data_features = pd.concat([data_features, data['IS_PAY_MODE_CONTAINS_020261']], axis=1)

        if self.is_rcvbl_amt_t_pq_equal0:
            print('RCVBL_AMT_T_PQ_equal0 1')
            data_features = pd.concat([data_features, data['RCVBL_AMT_T_PQ_equal0']], axis=1)

        if self.is_rcvbl_amt_equal0_t_pq_grater0:
            print('RCVBL_AMT_equal0_T_PQ_grater0 1')
            data_features = pd.concat([data_features, data['RCVBL_AMT_equal0_T_PQ_grater0']], axis=1)

        if self.is_rcvbl_amt_lower0:
            print('RCVBL_AMT_lower0 1')
            data_features = pd.concat([data_features, data['RCVBL_AMT_lower0']], axis=1)

        if self.is_penalty:
            print('是否违约 1')
            data['IS_PENALTY'] = data['NUM_OF_RCVBL_PENALTY'] > 0
            data_features = pd.concat([data_features, data['IS_PENALTY']], axis=1)

        if self.is_exceeding_rcvbl_ym_ge_1mon:
            print('是否违约超出1个月以上 1')
            data_features = pd.concat([data_features, data['IS_EXCEEDING_RCVBL_YM_GE_1MON']], axis=1)

        if self.is_hebiao_user_09table:
            print('IS_HEBIAO_USER 1')
            data_features = pd.concat([data_features, data['IS_HEBIAO_USER']], axis=1)
        if self.is_elec_eq_zero_09table:
            print('IS_ELEC_EQ_ZERO 1')
            data_features = pd.concat([data_features, data['IS_ELEC_EQ_ZERO']], axis=1)
        if self.is_seperate_time_09table:
            print('IS_SEPERATE_TIME 1')
            data_features = pd.concat([data_features, data['IS_SEPERATE_TIME']], axis=1)
        if self.is_mid_change_09table:
            print('IS_MID_CHANGE 1')
            data_features = pd.concat([data_features, data['IS_MID_CHANGE']], axis=1)

        if self.handle_year:
            # 年
            print('handle_year')
            feature_handle_year = data['HANDLE_TIME'].apply(lambda x: self.get_date(x, type='year'))
            data_features = pd.concat([data_features, feature_handle_year], axis=1)

        if self.handle_month:
            # 月
            print('代表工单月份 1')
            if 'HANDLE_MONTH' in data.columns:
                data_features = pd.concat([data_features, data['HANDLE_MONTH']], axis=1)
            else:
                feature_handle_month = data['HANDLE_TIME'].apply(lambda x: self.get_date(x, type='month'))
                data_features = pd.concat([data_features, feature_handle_month], axis=1)

        if self.handle_day:
            # 日
            print('handle_day 1')
            feature_handle_day = data['HANDLE_TIME'].apply(lambda x: self.get_date(x, type='day'))
            data_features = pd.concat([data_features, feature_handle_day], axis=1)

        if self.handle_hour:
            # 时
            print('handle_hour 1')
            feature_handle_hour = data['HANDLE_TIME'].apply(lambda x: self.get_date(x, type='hour'))
            data_features = pd.concat([data_features, feature_handle_hour], axis=1)

        if self.elec_degree_09table:
            print('电价到达档数 1')
            data_features = pd.concat([data_features, data['ELEC_DEGREE']], axis=1)

        # endregion
        # region 需要 onehot 编码
        if self.cust_no_3bit:
            print('CUST_NO前3位 %d' % len(self.cust_no_3bit_list))
            onehot_cust_no_3bit = self.get_onehot_encoding(data,
                                                           'CUST_NO_3bit',
                                                           labels=self.cust_no_3bit_list,
                                                           )
            data_features = pd.concat([data_features, onehot_cust_no_3bit], axis=1)

        if self.busi_type_code:
            print('业务类型编码(大工单类型) %d' % len(self.busi_type_code_list))
            onehot_busi_type_code = self.get_onehot_encoding(data,
                                                             'BUSI_TYPE_CODE',
                                                             labels=self.busi_type_code_list,
                                                             )
            data_features = pd.concat([data_features, onehot_busi_type_code], axis=1)
        if self.urban_rural_flag:
            print('城乡标志 %d' % len(self.urban_rural_flag_list))
            onehot_urban_rural_flag = self.get_onehot_encoding(data,
                                                               'URBAN_RURAL_FLAG',
                                                               labels=self.urban_rural_flag_list,
                                                               )
            data_features = pd.concat([data_features, onehot_urban_rural_flag], axis=1)
        if self.city_org_no:
            print('城市编码 %d' % len(self.city_org_no_list))
            onehot_city_org_no = self.get_onehot_encoding(data,
                                                          'CITY_ORG_NO',
                                                          labels=self.city_org_no_list,
                                                          )
            data_features = pd.concat([data_features, onehot_city_org_no], axis=1)
        if self.elec_type:
            print('用电类型 %d' % len(self.elec_type_list))
            onehot_elec_type = self.get_onehot_encoding(data,
                                                        'ELEC_TYPE',
                                                        labels=self.elec_type_list,
                                                        )
            data_features = pd.concat([data_features, onehot_elec_type], axis=1)
        if self.accept_content_type:
            print('小工单类型（single） %d' % len(self.accept_content_type_list))
            onehot_accept_content_type = self.get_onehot_encoding(data,
                                                                  'ACCEPT_CONTENT_TYPE',
                                                                  labels=self.accept_content_type_list,
                                                                  )
            data_features = pd.concat([data_features, onehot_accept_content_type], axis=1)
        if self.multi_accept_content_type:
            print('小工单类型(multi) %d' % len(self.multi_accept_content_type_list))
            onehot_multi_accept_content_type = self.get_onehot_encoding(data,
                                                                        'MULTI_ACCEPT_CONTENT_TYPE',
                                                                        labels=self.multi_accept_content_type_list,
                                                                        )
            data_features = pd.concat([data_features, onehot_multi_accept_content_type], axis=1)

        if self.cont_type_06table:
            print('CONT_TYPE %d' % len(self.cont_type_06table_list))
            onehot_cont_type = self.get_onehot_encoding(data,
                                                        'CONT_TYPE',
                                                        labels=self.cont_type_06table_list,
                                                        )
            data_features = pd.concat([data_features, onehot_cont_type], axis=1)

        if self.status_06table:
            print('STATUS %d' % len(self.status_06table_list))
            onehot_status = self.get_onehot_encoding(data,
                                                     'STATUS',
                                                     labels=self.status_06table_list,
                                                     )
            data_features = pd.concat([data_features, onehot_status], axis=1)

        if self.cons_status_07table:
            print('CONS_STATUS %d' % len(self.cons_status_07table_list))
            onehot_cons_status = self.get_onehot_encoding(data,
                                                          'CONS_STATUS',
                                                          labels=self.cons_status_07table_list,
                                                          )
            data_features = pd.concat([data_features, onehot_cons_status], axis=1)

        if self.rca_flag_07table:
            print('RCA_FLAG %d' % len(self.rca_flag_07table_list))
            onehot_rca_flag = self.get_onehot_encoding(data,
                                                       'RCA_FLAG',
                                                       labels=self.rca_flag_07table_list,
                                                       )
            data_features = pd.concat([data_features, onehot_rca_flag], axis=1)

        if self.org_no_7bit_09table:
            print('ORG_NO_7bit_09TABLE %d' % len(self.org_no_7bit_09table_list))
            onehot_org_no_7bit_09table = self.get_onehot_encoding(data,
                                                                  'ORG_NO_7bit',
                                                                  labels=self.org_no_7bit_09table_list,
                                                                  )
            data_features = pd.concat([data_features, onehot_org_no_7bit_09table], axis=1)
        if self.org_no_9bit_09table:
            print('ORG_NO_9bit_09TABLE %d' % len(self.org_no_9bit_09table_list))
            onehot_org_no_9bit_09table = self.get_onehot_encoding(data,
                                                                  'ORG_NO_9bit',
                                                                  labels=self.org_no_9bit_09table_list,
                                                                  )
            data_features = pd.concat([data_features, onehot_org_no_9bit_09table], axis=1)

        if self.last_month_pay_mode_09table:
            print('用户最后一个月的缴费方式 %d' % len(self.last_month_pay_mode_09table_list))
            onehot_last_month_pay_mode_09table = self.get_onehot_encoding(data,
                                                                          'LAST_MONTH_PAY_MODE',
                                                                          labels=self.last_month_pay_mode_09table_list,
                                                                          )
            data_features = pd.concat([data_features, onehot_last_month_pay_mode_09table], axis=1)

        if self.last_month_pay_mode_4bit_09table:
            print('用户最后一个月的缴费方式(4bit) %d' % len(self.last_month_pay_mode_4bit_09table_list))
            onehot_last_month_pay_mode_4bit_09table = self.get_onehot_encoding(data,
                                                                               'LAST_MONTH_PAY_MODE_4bit',
                                                                               labels=self.last_month_pay_mode_4bit_09table_list,
                                                                               )
            data_features = pd.concat([data_features, onehot_last_month_pay_mode_4bit_09table], axis=1)

        if self.pay_mode_change_clue_09table:
            print('用户缴费方式 的 转变路线 %d' % len(self.pay_mode_change_clue_09table_list))
            onehot_pay_mode_change_clue_09table = self.get_onehot_encoding(data,
                                                                           'PAY_MODE_CHANGE_CLUE',
                                                                           labels=self.pay_mode_change_clue_09table_list,
                                                                           )
            data_features = pd.concat([data_features, onehot_pay_mode_change_clue_09table], axis=1)

        if self.pay_mode_4bit_change_clue_09table:
            print('用户缴费方式(4bit) 的 转变路线 %d' % len(self.pay_mode_4bit_change_clue_09table_list))
            onehot_pay_mode_4bit_change_clue_09table = self.get_onehot_encoding(data,
                                                                                'PAY_MODE_4bit_CHANGE_CLUE',
                                                                                labels=self.pay_mode_4bit_change_clue_09table_list,
                                                                                )
            data_features = pd.concat([data_features, onehot_pay_mode_4bit_change_clue_09table], axis=1)
        # endregion

        return data_features

    def fit_transform(self, data):

        return self.fit(data).transform(data)

    def get_onehot_encoding(self, data, attribute_name, labels, mulit_type=False, join_str='+'):
        """ 提供 labels，将数据的某个属性进行onehot编码

        :param mulit_type: 是否一个记录里有多个 记录
        :param join_str: 当 mulit_type = True时，记录的连接字符
        :param data:
        :param attribute_name:
        :param labels:
        :return:
        """
        assert (data is not None) or (attribute_name is not None), 'data is None or attribute_name is None!'
        assert attribute_name in data.columns, 'attribute_name not in data!'
        data_onehot = pd.DataFrame()
        for index, value in enumerate(labels):
            if not mulit_type:
                data_onehot['%s_%s' % (attribute_name, value)] = data[attribute_name] == value
            else:
                data_onehot['%s_%s' % (attribute_name, value)] = map(lambda x: x is not np.nan and value in x,
                                                                     data[attribute_name].str.split(
                                                                         join_str).as_matrix())

        return data_onehot.astype(int)

    def get_date(self, x, type='hour'):
        """
        # 将受理时间进行编码，映射成 24个小时，0-23
                # 0 - 24
                # 1 - 1
                # 2 - 2
                # ...
                # 23 - 23
        :param type: 类型 ，'year','month','day','hour'
        :param x:
        :return:
        """
        assert type in ['year', 'month', 'day', 'hour'], 'type not in [year,month,day,hour]!'

        try:
            if type == 'year':
                return datetime.strptime(x, '%Y/%m/%d %H:%M:%S').year
            if type == 'month':
                return datetime.strptime(x, '%Y/%m/%d %H:%M:%S').month
            if type == 'day':
                return datetime.strptime(x, '%Y/%m/%d %H:%M:%S').day
            if type == 'hour':
                return datetime.strptime(x, '%Y/%m/%d %H:%M:%S').hour
        except:
            # 2015/8/15
            # 没有时间，只有日期
            if type == 'year':
                return datetime.strptime(x, '%Y/%m/%d').year
            if type == 'month':
                return datetime.strptime(x, '%Y/%m/%d').month
            if type == 'day':
                return datetime.strptime(x, '%Y/%m/%d').day
            if type == 'hour':
                return datetime.strptime(x, '%Y/%m/%d').hour

    def get_month(self, x):
        """
        # 将受理时间进行编码，映射成 月
        :param x:
        :return:
        """
        try:
            return datetime.strptime(x, '%Y/%m/%d %H:%M:%S').month
        except:
            # 2015/8/15
            # 没有时间，只有日期
            return datetime.strptime(x, '%Y/%m/%d').month

    def get_day(self, x):
        """
        # 将受理时间进行编码，映射成 日
        :param x:
        :return:
        """
        try:
            return datetime.strptime(x, '%Y/%m/%d %H:%M:%S').day
        except:
            # 2015/8/15
            # 没有时间，只有日期
            return datetime.strptime(x, '%Y/%m/%d').day


if __name__ == '__main__':
    features = FeatureEncoder()
