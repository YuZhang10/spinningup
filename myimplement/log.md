## PPO经验
- 出错一定要及时reset
我之前用固定长度的buffer，采集trajectory。发现其中相当一部分的数据都是0，0，0……这些无意义的state来回转移，本质上没有价值；但是被取出成为mini_batch供神经网络学习，污染了神经网络的数据，使其越来越差

- 分清楚old和new
pi-loss：
    - At，是根据历史数据计算出来，没有任何需要当前网络重新predict的地方。使用rewards，以及历史预测的value（如果用到自举的话）
    - ratio，这个比率是由当前actor在历史at上的概率值，与过去的actor在历史at上的概率值之比
vf-loss：
    - return，是根据历史数据计算出来的return
    - value，当前的critic给出当前state下的价值

- 测试时action用最大概率的
train时是按分布随机sample出action，但是在test时，应该用分布中最大的那个。
记得调换.eval()和.train()

- Tensor运算时的维度
一个(32,1)的Tensor -  （32）的Tensor，结果竟然是（32，32）的Tensor
所以为了保险，建议使用reshape_as函数

- PPO是on-policy：
每一次实现完之后，都需要之前存储的结果全部清空。所以每一次学习都是基于上一次交互的经验，而并非过去很久之前的经验。
