# customerPortrait
[kehu huaxiang](http://www.datafountain.cn/data/science/player/competition/detail/description/242)
思路：
- 挖掘显式特征和隐式特征;
- 统计 各类特征的分布情况 ： 

- Project 结构
    - 初赛：preliminary
    - 复赛：semi-finals
        - 程序
            - main.ipynb：主程序
            - helper 文件夹：数据工具、特征处理器、和部分特征提取脚本...
            - data_processing-01table.ipynb：表1特征的提取
            - data_processing-04&05&10&11&12table.ipynb：特征的提取
            - data_processing-06&07table.ipynb：特征的提取
            - data_processing-08table.ipynb：特征的提取
            - data_processing-09table.ipynb：特征的提取
            - data_processing-old.ipynb： old version...
        - 统计汇总：
            - 统计汇总/(含线下F值计算)train&test_accept_content_type.xlsx
        
        - 结果：
            - 本地验证结果和测试结果
            - 最终线上排名：
                - 初赛：A榜 - 0.22992，排第52; B榜 - 0.22722，排第53;
                - 复赛：A榜 - 0.70404，排第9; B榜 - 0.70925，排第8;
        - 数据集：
          - Data_Update： 修正后的比赛数据集  ----> 为节省空间，已进行压缩和[存盘]()，可以直接下载压缩到  semi-finals 文件夹即可
          - data_temp：数据缓存  ----> 为节省空间，已进行压缩和[存盘]()，可以直接下载压缩到  semi-finals 文件夹即可
            - test_data01_a_worker_per_user.csv： 测试集特征
            - train_data01_a_worker_per_user.csv： 训练集特征
            - test_data09_merge_label_df.csv： 表9测试集特征
            - train_data09_merge_label_df.csv： 表9训练集特征
            
- 读取文件时候，要特别注意 各个字段的类型，如果是字符串类型，被以整型读取将会改变不少，比如损失很多0等
    - 可以通过设置 converters={'CONS_NO':unicode} 等解决
    

- Tip:
    - 可以使用 ipython 来 缓存变量，便于不同核 的交流，以及下次的变量恢复。
    
    - pandas 的 groupby()函数非常好用。
    
    - pandas 的plot 函数画图不错
    
    - 多利用excel来整理和记录数据