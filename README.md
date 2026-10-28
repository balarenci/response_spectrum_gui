# response_spectrum_gui
read K-NET ASCII seismic data to calculate the response spectrum

# 输入
  * miniSEED (如 TitanSMA 记录, 单位 counts, 默认 2 counts/µg)
  * K-NET / KiK-net ASCII (.EW .NS .UD .EW1 ... , 比例因子从文件头自动读取);
    可直接选 NIED 下载的 .tar.gz 压缩包或文件夹, 多台站时弹窗选择台站
  * 可一次选多个文件/多个分量

# 计算
  * 单自由度振子, 阻尼比默认 5%, 周期范围与点数可设
  * Nigam-Jennings (1969) 分段线性精确解法, 对任意采样间隔都无数值误差
  * 输出 PSA(伪谱加速度 = ω²·Sd, 工程常用) 和 SA(绝对加速度最大值),
    以及 SV/SD; 可显示水平两分量几何平均
  * 叠加 GB 50011-2010(2016年版) 设计反应谱对比 (烈度/地震水准/场地类别/
    设计地震分组可选, 按所设阻尼比修正)

# 运行
python response_spectrum_gui.py

# 需要
pip install obspy numpy scipy matplotlib
