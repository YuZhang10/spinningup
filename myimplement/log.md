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