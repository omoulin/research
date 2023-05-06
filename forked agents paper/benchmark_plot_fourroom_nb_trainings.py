import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

eco = np.array([26,408,800,1104,1534,1776,2076,2384,2728,2854,3114])
eco2 = np.array([14,516,794,1044,1302,1704,1960,2188,2420,2702,2938])
eco3 = np.array([20,532,848,1094,1372,1640,2032,2254,2446,2622,2770])
eco4 = np.array([16,340,610,1024,1344,1620,1840,2060,2400,2750,2956])
eco5 = np.array([10,358,648,1166,1392,1576,1876,2088,2328,2514,2866])

min_eco = np.minimum(eco,eco2)
min_eco = np.minimum(min_eco,eco3)
min_eco = np.minimum(min_eco,eco4)
min_eco = np.minimum(min_eco,eco5)

max_eco = np.maximum(eco,eco2)
max_eco = np.maximum(max_eco,eco3)
max_eco = np.maximum(max_eco,eco4)
max_eco = np.maximum(max_eco,eco5)

avg_eco = eco+eco2+eco3+eco4+eco5
avg_eco = avg_eco /5

eco_var = eco - avg_eco
eco_var = np.square(eco_var)
eco_var2 = eco2 - avg_eco
eco_var2 = np.square(eco_var2)
eco_var3 = eco3 - avg_eco
eco_var3 = np.square(eco_var3)
eco_var4 = eco4 - avg_eco
eco_var4 = np.square(eco_var4)
eco_var5 = eco5 - avg_eco
eco_var5 = np.square(eco_var5)

eco_std_dev = eco_var + eco_var2 + eco_var3 + eco_var4 + eco_var5
eco_std_dev = eco_std_dev/5
eco_std_dev = np.sqrt(eco_std_dev)
eco_std_error = eco_std_dev / math.sqrt(5)


eco_best_init = np.array([8,186,310,443,583,710,850,948,1058,1190,1295])
eco_best_init2 = np.array([10,243,398,500,597,684,782,907,1057,1211,1308])
eco_best_init3 = np.array([7,194,346,529,644,759,899,1070,1192,1261,1319])
eco_best_init4 = np.array([9,238,381,557,653,825,933,1040,1153,1221,1335])
eco_best_init5 = np.array([4,315,505,623,706,862,950,1028,1099,1202,1308])

min_eco_best_init = np.minimum(eco_best_init,eco_best_init2)
min_eco_best_init = np.minimum(min_eco_best_init,eco_best_init3)
min_eco_best_init = np.minimum(min_eco_best_init,eco_best_init4)
min_eco_best_init = np.minimum(min_eco_best_init,eco_best_init5)

max_eco_best_init = np.maximum(eco_best_init,eco_best_init2)
max_eco_best_init = np.maximum(max_eco_best_init,eco_best_init3)
max_eco_best_init = np.maximum(max_eco_best_init,eco_best_init4)
max_eco_best_init = np.maximum(max_eco_best_init,eco_best_init5)

avg_eco_best_init = eco_best_init+eco_best_init2+eco_best_init3+eco_best_init4+eco_best_init5
avg_eco_best_init = avg_eco_best_init /5

eco_best_init_var = eco_best_init - avg_eco_best_init
eco_best_init_var = np.square(eco_best_init_var)
eco_best_init_var2 = eco_best_init2 - avg_eco_best_init
eco_best_init_var2 = np.square(eco_best_init_var2)
eco_best_init_var3 = eco_best_init3 - avg_eco_best_init
eco_best_init_var3 = np.square(eco_best_init_var3)
eco_best_init_var4 = eco_best_init4 - avg_eco_best_init
eco_best_init_var4 = np.square(eco_best_init_var4)
eco_best_init_var5 = eco_best_init5 - avg_eco_best_init
eco_best_init_var5 = np.square(eco_best_init_var5)

eco_best_init_std_dev = eco_best_init_var + eco_best_init_var2 + eco_best_init_var3 + eco_best_init_var4 + eco_best_init_var5
eco_best_init_std_dev = eco_best_init_std_dev/5
eco_best_init_std_dev = np.sqrt(eco_best_init_std_dev)
eco_best_init_std_error = eco_best_init_std_dev / math.sqrt(5)

eco_one_init = np.array([7,167,233,298,377,429,521,594,661,710,754])
eco_one_init2 = np.array([9,86,203,318,385,464,509,584,679,738,800])
eco_one_init3 = np.array([11,110,213,296,414,483,546,608,659,696,742])
eco_one_init4 = np.array([14,156,235,323,394,468,571,695,780,885,983])
eco_one_init5 = np.array([6,109,216,324,384,448,542,615,657,728,771])

min_eco_one_init = np.minimum(eco_one_init,eco_one_init2)
min_eco_one_init = np.minimum(min_eco_one_init,eco_one_init3)
min_eco_one_init = np.minimum(min_eco_one_init,eco_one_init4)
min_eco_one_init = np.minimum(min_eco_one_init,eco_one_init5)

max_eco_one_init = np.maximum(eco_one_init,eco_one_init2)
max_eco_one_init = np.maximum(max_eco_one_init,eco_one_init3)
max_eco_one_init = np.maximum(max_eco_one_init,eco_one_init4)
max_eco_one_init = np.maximum(max_eco_one_init,eco_one_init5)

avg_eco_one_init = eco_one_init+eco_one_init2+eco_one_init3+eco_one_init4+eco_one_init5
avg_eco_one_init = avg_eco_one_init /5

eco_one_init_var = eco_one_init - avg_eco_one_init
eco_one_init_var = np.square(eco_one_init_var)
eco_one_init_var2 = eco_one_init2 - avg_eco_one_init
eco_one_init_var2 = np.square(eco_one_init_var2)
eco_one_init_var3 = eco_one_init3 - avg_eco_one_init
eco_one_init_var3 = np.square(eco_one_init_var3)
eco_one_init_var4 = eco_one_init4 - avg_eco_one_init
eco_one_init_var4 = np.square(eco_one_init_var4)
eco_one_init_var5 = eco_one_init5 - avg_eco_one_init
eco_one_init_var5 = np.square(eco_one_init_var5)

eco_one_init_std_dev = eco_one_init_var + eco_one_init_var2 + eco_one_init_var3 + eco_one_init_var4 + eco_one_init_var5
eco_one_init_std_dev = eco_one_init_std_dev/5
eco_one_init_std_dev = np.sqrt(eco_one_init_std_dev)
eco_one_init_std_error = eco_one_init_std_dev / math.sqrt(5)

eco_rnd_init = np.array([17,143,244,377,484,618,749,838,944,1042,1146])
eco_rnd_init2 = np.array([15,142,291,458,530,684,798,899,964,1096,1237])
eco_rnd_init3 = np.array([10,175,318,395,460,574,681,764,913,1011,1078])
eco_rnd_init4 = np.array([8,133,247,367,469,615,684,772,850,988,1087])
eco_rnd_init5 = np.array([27,153,302,423,526,592,689,812,924,1016,1105])

min_eco_rnd_init = np.minimum(eco_rnd_init,eco_rnd_init2)
min_eco_rnd_init = np.minimum(min_eco_rnd_init,eco_rnd_init3)
min_eco_rnd_init = np.minimum(min_eco_rnd_init,eco_rnd_init4)
min_eco_rnd_init = np.minimum(min_eco_rnd_init,eco_rnd_init5)

max_eco_rnd_init = np.maximum(eco_rnd_init,eco_rnd_init2)
max_eco_rnd_init = np.maximum(max_eco_rnd_init,eco_rnd_init3)
max_eco_rnd_init = np.maximum(max_eco_rnd_init,eco_rnd_init4)
max_eco_rnd_init = np.maximum(max_eco_rnd_init,eco_rnd_init5)

avg_eco_rnd_init = eco_rnd_init+eco_rnd_init2+eco_rnd_init3+eco_rnd_init4+eco_rnd_init5
avg_eco_rnd_init = avg_eco_rnd_init /5

eco_rnd_init_var = eco_rnd_init - avg_eco_rnd_init
eco_rnd_init_var = np.square(eco_rnd_init_var)
eco_rnd_init_var2 = eco_rnd_init2 - avg_eco_rnd_init
eco_rnd_init_var2 = np.square(eco_rnd_init_var2)
eco_rnd_init_var3 = eco_rnd_init3 - avg_eco_rnd_init
eco_rnd_init_var3 = np.square(eco_rnd_init_var3)
eco_rnd_init_var4 = eco_rnd_init4 - avg_eco_rnd_init
eco_rnd_init_var4 = np.square(eco_rnd_init_var4)
eco_rnd_init_var5 = eco_rnd_init5 - avg_eco_rnd_init
eco_rnd_init_var5 = np.square(eco_rnd_init_var5)

eco_rnd_init_std_dev = eco_rnd_init_var + eco_rnd_init_var2 + eco_rnd_init_var3 + eco_rnd_init_var4 + eco_rnd_init_var5
eco_rnd_init_std_dev = eco_rnd_init_std_dev/5
eco_rnd_init_std_dev = np.sqrt(eco_rnd_init_std_dev)
eco_rnd_init_std_error = eco_rnd_init_std_dev / math.sqrt(5)

x=[0,1,2,3,4,5,6,7,8,9,10]
plt.title("Training cycles needed", size=15,y=1.06)
plt.xticks([0,2,4,6,8,10],[1,100,200,300,400,500],rotation=0)
plt.ylabel('# of steps (x50000)', size=14)
plt.xlabel('# of environment trained',size =14)  
plt.plot(avg_eco,color='purple',label='Eco-system')
plt.fill_between(x, avg_eco - eco_std_error, avg_eco + eco_std_error,facecolor="purple",color='purple',alpha=0.2)  

plt.plot(avg_eco_best_init,color='orange',label='Eco-system best agent init.')
plt.fill_between(x, avg_eco_best_init - eco_best_init_std_error, avg_eco_best_init + eco_best_init_std_error,facecolor="orange",color='orange',alpha=0.2)  

plt.plot(avg_eco_one_init,color='green',label='Eco-system one agent init.')
plt.fill_between(x, avg_eco_one_init - eco_one_init_std_error, avg_eco_one_init + eco_one_init_std_error,facecolor="green",color='green',alpha=0.2)  

plt.plot(avg_eco_rnd_init,color='lightblue',label='Eco-system random agent init.')
plt.fill_between(x, avg_eco_rnd_init - eco_rnd_init_std_error, avg_eco_rnd_init + eco_rnd_init_std_error,facecolor="lightblue",color='lightblue',alpha=0.2)  

orange_patch = mpatches.Patch(color='orange', label='best agent init.')
purple_patch = mpatches.Patch(color='purple', label='no init.')
green_patch = mpatches.Patch(color='green', label='forked agent init.')
blue_patch = mpatches.Patch(color='lightblue', label='random agent init.')

plt.legend(handles=[purple_patch,orange_patch,green_patch,blue_patch])
plt.savefig('Benchmark_fourroom_nb_steps.pdf')
plt.close()