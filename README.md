# customerPortrait
kehu huaxiang
思路：
- 挖掘显式特征和隐式特征;
- 统计 各类特征的分布情况 ： 


- 读取文件时候，要特别注意 各个字段的类型，如果是字符串类型，被以整型读取将会改变不少，比如损失很多0等
    - 可以通过设置 converters={'CONS_NO':unicode} 等解决