# **同步度计算**
## **同步图**
- 在较慢波中找出较快波峰值出现时刻的相位值
- 同步图示例

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
# 参数设置
omega_x = 1
omega_y = 5
t = np.arange(0, 100, 0.01)  
x = np.sin(omega_x * np.pi*t)  
y = np.sin(omega_y * np.pi*t) 

# 计算较慢波（x）的相位，使用 Hilbert 变换
phi_x = np.angle(hilbert(x))

# 找到较快波（y）的峰值位置
peaks, _ = find_peaks(y)

# 在较快波的峰值位置获取较慢波的相位值
sync = phi_x[peaks]

# 绘图
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# 绘制信号图
axes[0].plot(t, x, 'b', label='x(slow)')
axes[0].plot(t, y, 'r', label='y(fast)')
axes[0].legend()

# 绘制相位同步图（Synchrogram）
axes[1].plot(peaks * 0.01, sync, 'o')  # peaks 是索引，乘以 0.01 转换为时间值

plt.tight_layout()
plt.show()
```
## **同步度计算步骤**
- 参考[Cardiorespiratory Phase Synchronization in Elderly Patients with 
Periodic and non-Periodic Breathing Patterns](https://pubmed.ncbi.nlm.nih.gov/36086581/)
- 要计算某段心脏呼吸信号的同步度，首先得到呼吸信号的相位和心跳时刻
- 归一化相对相位：计算心跳时刻的呼吸相位并约束在0~2* pi* m区间，再乘以1/(2* pi)进行相位归一化
- 将n条水平线合并成一条水平线，滑窗计算，在窗口内使用类似圆方差的方法计算同步度
# **同步图函数工作流程**
- 使用希尔伯特变换得到呼吸相位相位，范围为-pi~ pi，进行相位展开，再使用mod操作将相位约束在0~ 2* pi* m之间，以查看m个呼吸周期内的心跳情况
- 信号处理：对信号进行带通滤波
- 使用20s的窗口计算每个心跳时刻对应的第一个猜测的n:1并记录下来
- 使用大小为N=50的窗口计算每个点（心跳时刻）的所有可能的n:m所对应的同步度，选取其中同步度最大的作为该点的同步度，记录相应的同步比
- 确定同步度阈值，计算相应的评价指标：
- - **%Sync**： the percentage of synchronization defined as the 
sum of all the synchronization periods over the recording.
- - **AvgDurSync**: the average duration of the synchronization 
as the arithmetic mean of all periods of synchronization 
over the duration of the segment.
- - **NumSync**: the number of synchronized epochs defined as 
the sum of synchronization periods over the duration of 
the segment.
- - **FreqRat**:the frequency ratio between the respiratory and 
cardiac oscillators, as an approximation to the number of 
heartbeats per each respiration.
- 绘制同步度的变化图
- 绘制同步比的频率热图
- 绘制同步比随时间变化的折线图
## **检测所有可能的n:m**
- 参考论文[Mechanical ventilatory modes and cardioventilatory phase synchronization in acute respiratory failure patients](https://iopscience.iop.org/article/10.1088/1361-6579/aa56ae)的方法：
初步确定一个n1:1，可以使用心跳频率和呼吸频率的比值获得，如比值为3.27：1，则n:m的第一个猜测为3：1
- 在 m = 1 时，该过程将最接近猜测的两个比率（即 （n1− 1）：1 和 （n1+ 1）:1）。这两个比率称为最左和最右的比率。例如，最左侧和最右侧的比率为 m = 1 时为 2：1 和 4：1。接着 2：1 和 4：1 以 4：2 和 8：2 的形式传播到 m = 2，以 6：3 和 12：3 的形式传播到 m = 3。给定 m，则传播的比率（包括极值）都被接受为候选值。
