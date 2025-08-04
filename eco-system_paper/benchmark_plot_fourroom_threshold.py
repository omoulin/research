import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

eco_8 = np.array([0.13,0.61,0.69,0.68]) #79
eco2_8 = np.array([0.18,0.59,0.64,0.70])
eco3_8 = np.array([0.25,0.64,0.75,0.70])
eco4_8 = np.array([0.09,0.65,0.70,0.74])
eco5_8 = np.array([0.23,0.62,0.75,0.72])

min_eco_8 = np.minimum(eco_8,eco2_8)
min_eco_8 = np.minimum(min_eco_8,eco3_8)
min_eco_8 = np.minimum(min_eco_8,eco4_8)
min_eco_8 = np.minimum(min_eco_8,eco5_8)

max_eco_8 = np.maximum(eco_8,eco2_8)
max_eco_8 = np.maximum(max_eco_8,eco3_8)
max_eco_8 = np.maximum(max_eco_8,eco4_8)
max_eco_8 = np.maximum(max_eco_8,eco5_8)

avg_eco_8 = eco_8+eco2_8+eco3_8+eco4_8+eco5_8
avg_eco_8 = avg_eco_8 /5

eco_var_8 = eco_8 - avg_eco_8
eco_var_8 = np.square(eco_var_8)
eco_var2_8 = eco2_8 - avg_eco_8
eco_var2_8 = np.square(eco_var2_8)
eco_var3_8 = eco3_8 - avg_eco_8
eco_var3_8 = np.square(eco_var3_8)
eco_var4_8 = eco4_8 - avg_eco_8
eco_var4_8 = np.square(eco_var4_8)
eco_var5_8 = eco5_8 - avg_eco_8
eco_var5_8 = np.square(eco_var5_8)

eco_std_dev_8 = eco_var_8 + eco_var2_8 + eco_var3_8 + eco_var4_8 + eco_var5_8
eco_std_dev_8 = eco_std_dev_8/5
eco_std_dev_8 = np.sqrt(eco_std_dev_8)

eco_85 = np.array([0.19,0.72,0.73,0.76]) #89
eco2_85 = np.array([0.15,0.66,0.75,0.75]) #95
eco3_85 = np.array([0.13,0.59,0.73,0.78]) #84
eco4_85 = np.array([0.27,0.65,0.71,0.76]) #93
eco5_85 = np.array([0.15,0.65,0.71,0.76]) #93

min_eco_85 = np.minimum(eco_85,eco2_85)
min_eco_85 = np.minimum(min_eco_85,eco3_85)
min_eco_85 = np.minimum(min_eco_85,eco4_85)
min_eco_85 = np.minimum(min_eco_85,eco5_85)

max_eco_85 = np.maximum(eco_85,eco2_85)
max_eco_85 = np.maximum(max_eco_85,eco3_85)
max_eco_85 = np.maximum(max_eco_85,eco4_85)
max_eco_85 = np.maximum(max_eco_85,eco5_85)

avg_eco_85 = eco_85+eco2_85+eco3_85+eco4_85+eco5_85
avg_eco_85 = avg_eco_85 /5

eco_var_85 = eco_85 - avg_eco_85
eco_var_85 = np.square(eco_var_85)
eco_var2_85 = eco2_85 - avg_eco_85
eco_var2_85 = np.square(eco_var2_85)
eco_var3_85 = eco3_85 - avg_eco_85
eco_var3_85 = np.square(eco_var3_85)
eco_var4_85 = eco4_85 - avg_eco_85
eco_var4_85 = np.square(eco_var4_85)
eco_var5_85 = eco5_85 - avg_eco_85
eco_var5_85 = np.square(eco_var5_85)

eco_std_dev_85 = eco_var_85 + eco_var2_85 + eco_var3_85 + eco_var4_85 + eco_var5_85
eco_std_dev_85 = eco_std_dev_85/5
eco_std_dev_85 = np.sqrt(eco_std_dev_85)

eco_9 = np.array([0.15,0.77,0.79,0.77]) #115
eco2_9 = np.array([0.24,0.71,0.77,0.75]) #101
eco3_9 = np.array([0.24,0.76,0.75,0.73]) #102
eco4_9 = np.array([0.23,0.66,0.74,0.76]) #102
eco5_9 = np.array([0.30,0.71,0.75,0.76]) #115

min_eco_9 = np.minimum(eco_9,eco2_9)
min_eco_9 = np.minimum(min_eco_9,eco3_9)
min_eco_9 = np.minimum(min_eco_9,eco4_9)
min_eco_9 = np.minimum(min_eco_9,eco5_9)

max_eco_9 = np.maximum(eco_9,eco2_9)
max_eco_9 = np.maximum(max_eco_9,eco3_9)
max_eco_9 = np.maximum(max_eco_9,eco4_9)
max_eco_9 = np.maximum(max_eco_9,eco5_9)

avg_eco_9 = eco_9+eco2_9+eco3_9+eco4_9+eco5_9
avg_eco_9 = avg_eco_9 /5

eco_var_9 = eco_9 - avg_eco_9
eco_var_9 = np.square(eco_var_9)
eco_var2_9 = eco2_9 - avg_eco_9
eco_var2_9 = np.square(eco_var2_9)
eco_var3_9 = eco3_9 - avg_eco_9
eco_var3_9 = np.square(eco_var3_9)
eco_var4_9 = eco4_9 - avg_eco_9
eco_var4_9 = np.square(eco_var4_9)
eco_var5_9 = eco5_9 - avg_eco_9
eco_var5_9 = np.square(eco_var5_9)

eco_std_dev_9 = eco_var_9 + eco_var2_9 + eco_var3_9 + eco_var4_9 + eco_var5_9
eco_std_dev_9 = eco_std_dev_9/5
eco_std_dev_9 = np.sqrt(eco_std_dev_9)

x=[0,1,2,3]
plt.title("Generalizability index on new environments", size=16,y=1.06)
plt.xticks([0,1,2,3],[5,150,402,564],rotation=0,size=15)
plt.yticks(size=15)
plt.ylabel('Avg. expected return', size=16)
plt.xlabel('# of training steps (x100000)',size =14)  

plt.plot(avg_eco_8,color='green',label='Threshold = 0.8')
plt.fill_between(x, avg_eco_8 - eco_std_dev_8, avg_eco_8 + eco_std_dev_8,facecolor="green",color='green',alpha=0.2)  

plt.plot(avg_eco_85,color='purple',label='Threshold = 0.85')
plt.fill_between(x, avg_eco_85 - eco_std_dev_85, avg_eco_85 + eco_std_dev_85,facecolor="purple",color='purple',alpha=0.2)  

plt.plot(avg_eco_9,color='orange',label='Threshold = 0.9')
plt.fill_between(x, avg_eco_9 - eco_std_dev_9, avg_eco_9 + eco_std_dev_9,facecolor="orange",color='orange',alpha=0.2)  

red_patch = mpatches.Patch(color='orange', label='Threshold = 0.9')
blue_patch = mpatches.Patch(color='purple', label='Threshold = 0.85')
green_patch = mpatches.Patch(color='green', label='Threshold = 0.8')
plt.legend(handles=[red_patch,blue_patch,green_patch])
plt.savefig('Benchmark_fourroom_threshold.pdf')
plt.close()
