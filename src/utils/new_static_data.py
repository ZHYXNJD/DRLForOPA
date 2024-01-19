import numpy as np

#-----------------------------------停车场分布------------------------------


parking_lot_num = 3
destination_num = 3

req_info_path = r"G:\停车数据\原始数据备份\11月份处理\1206-全部信息"
r_mk_path = r"G:\停车数据\原始数据备份\11月份处理\1206-rmk"
charge_fee_path = r"G:\停车数据\原始数据备份\11月份处理\1128_charge_fee"

# 设计停车相关的矩阵
# parking lot 1: 100 OPS  15 CPS  115
# parking lot 2: 120 OPS  20 CPS  140
# parking lot : 60 OPS  8 CPS    68
# total OPS 280 ; total CPS 43 ; total slots 323;
pl1_total_slots = 100
pl1_total_piles = 15
pl1_total = pl1_total_slots + pl1_total_piles

pl2_total_slots = 120
pl2_total_piles = 20
pl2_total = pl2_total_slots + pl2_total_piles


pl3_total_slots = 60
pl3_total_piles = 8
pl3_total = pl3_total_slots + pl3_total_piles

total_slot = pl1_total + pl2_total + pl3_total  # 所有停车场总的车位数+充电桩数量
total_slots = pl3_total_slots+pl2_total_slots+pl1_total_slots  # 所有停车场的总泊位数（不包括充电桩）
total_piles = pl1_total_piles + pl2_total_piles + pl3_total_piles  # 所有停车场的充电桩数量

# 普通泊位和充电桩泊位分布
B_ZN = np.zeros((parking_lot_num,total_slot))
B_ZN[0,:pl1_total] = 1
B_ZN[1,pl1_total:pl1_total+pl2_total] = 1
B_ZN[2,pl1_total+pl2_total:] = 1
# 普通泊位分布
P_ZN = np.zeros((parking_lot_num,total_slot))
P_ZN[0,:pl1_total_slots] = 1
P_ZN[1,pl1_total:pl1_total+pl2_total_slots] = 1
P_ZN[2,pl1_total+pl2_total:pl1_total+pl2_total+pl3_total_slots] = 1
# 充电桩泊位分布
C_ZN = np.zeros((parking_lot_num, total_slot))
C_ZN[0, pl1_total_slots:pl1_total] = 1
C_ZN[1, pl1_total+pl2_total_slots:pl1_total+pl2_total] = 1
C_ZN[2, pl1_total+pl2_total+pl3_total_slots:] = 1

N_Z = [pl1_total,pl2_total,pl3_total]
N_Z_char = [pl1_total_piles,pl2_total_piles,pl3_total_piles]
N_Z_park = [pl1_total_slots,pl2_total_slots,pl3_total_slots]

ops_index = list(range(pl1_total_slots)) + list(range(pl1_total,pl1_total+pl2_total_slots)) + list(range(pl1_total+pl2_total,pl1_total+pl2_total+pl3_total_slots))
cps_index = list(range(pl1_total_slots,pl1_total)) + list(range(pl1_total+pl2_total_slots,pl1_total+pl2_total)) + list(range(pl1_total+pl2_total+pl3_total_slots,total_slot))


#-------------------------------------------------------------------#
# 距离矩阵
L_ZD = np.zeros((parking_lot_num,destination_num)).astype(int)
L_ZD[0] = [580,590,600]
L_ZD[1] = [630,550,400]
L_ZD[2] = [620,380,560]

#------------------------------------------------------------#
#平台相关
N = total_slot
C_park = 1.5  # 停车费，元/15min
C_res = 1   # 预约费，元/人
# 国家电网收费

alpha1 = 0.5   # 平台收益
alpha2 = 0.2   # 拒绝请求惩罚
alpha3 = 0.3   # 用户步行时间
# alpha4 = 0.2   # 普通泊位利用率和充电桩利用率
w_2 = 0.01 # 单位：CNY/s，步行时间损耗
v_walk = 1.2 # 单位：m/s，步行速度



