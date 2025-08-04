import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 3,71,89,122,145,168,200,235,263,296,314,344,358,371,392,417,447,483,507,537,568,600,637

ibac = np.array([0.02,0.07,0.16,0.21,0.21,0.33,0.41,0.41,0.50,0.53,0.57,0.59,0.59,0.64,0.59,0.65,0.70,0.73,0.75,0.73,0.74,0.76,0.74])
ibac2 = np.array([0.01,0.02,0.01,0.05,0.13,0.14,0.21,0.24,0.30,0.32,0.36,0.45,0.50,0.53,0.59,0.58,0.60,0.66,0.73,0.74,0.75,0.77,0.76])
ibac3 = np.array([0.01,0.15,0.23,0.30,0.30,0.36,0.39,0.49,0.58,0.67,0.70,0.72,0.73,0.75,0.74,0.76,0.75,0.76,0.77,0.75,0.77,0.77,0.76])
ibac4 = np.array([0.01,0.15,0.23,0.65,0.69,0.71,0.72,0.73,0.76,0.76,0.76,0.74,0.76,0.73,0.78,0.75,0.75,0.76,0.76,0.75,0.77,0.77,0.77])
ibac5 = np.array([0.02,0.15,0.25,0.35,0.47,0.46,0.66,0.72,0.75,0.74,0.76,0.74,0.72,0.72,0.76,0.76,0.78,0.77,0.76,0.75,0.76,0.77,0.79])

min_ibac = np.minimum(ibac,ibac2)
min_ibac = np.minimum(min_ibac,ibac3)
min_ibac = np.minimum(min_ibac,ibac4)
min_ibac = np.minimum(min_ibac,ibac5)

max_ibac = np.maximum(ibac,ibac2)
max_ibac = np.maximum(max_ibac,ibac3)
max_ibac = np.maximum(max_ibac,ibac4)
max_ibac = np.maximum(max_ibac,ibac5)

avg_ibac = ibac+ibac2+ibac3+ibac4+ibac5
avg_ibac = avg_ibac /5

ibac_var = ibac - avg_ibac
ibac_var = np.square(ibac_var)
ibac_var2 = ibac2 - avg_ibac
ibac_var2 = np.square(ibac_var2)
ibac_var3 = ibac3 - avg_ibac
ibac_var3 = np.square(ibac_var3)
ibac_var4 = ibac4 - avg_ibac
ibac_var4 = np.square(ibac_var4)
ibac_var5 = ibac5 - avg_ibac
ibac_var5 = np.square(ibac_var5)

ibac_std_dev = ibac_var + ibac_var2 + ibac_var3 + ibac_var4 + ibac_var5
ibac_std_dev = ibac_std_dev/5
ibac_std_dev = np.sqrt(ibac_std_dev)

eco = np.array([0.79,0.82,0.85,0.83,0.84,0.84,0.85,0.83,0.84,0.85,0.84,0.82,0.83,0.85,0.83,0.85,0.84,0.84,0.84,0.85,0.84,0.83,0.84])
eco2 = np.array([0.77,0.84,0.84,0.84,0.84,0.84,0.85,0.83,0.85,0.83,0.83,0.83,0.84,0.85,0.84,0.84,0.84,0.84,0.86,0.85,0.83,0.83,0.83])
eco3 = np.array([0.82,0.83,0.85,0.83,0.83,0.84,0.84,0.84,0.83,0.83,0.82,0.82,0.83,0.85,0.84,0.85,0.84,0.83,0.84,0.84,0.83,0.84,0.84])
eco4 = np.array([0.79,0.84,0.84,0.85,0.83,0.84,0.84,0.84,0.85,0.84,0.85,0.85,0.84,0.83,0.86,0.84,0.85,0.84,0.83,0.84,0.83,0.84,0.84])
eco5 = np.array([0.83,0.83,0.82,0.83,0.83,0.84,0.83,0.86,0.84,0.85,0.85,0.84,0.83,0.84,0.85,0.85,0.84,0.84,0.84,0.83,0.84,0.83,0.84])

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

plt.title("Generalizability index on new environments", size=16,y=1.06)
x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
plt.xticks([0,4,8,12,16,20],[3,145,263,358,447,568],rotation=0,size=15)
plt.yticks(size=15)
plt.ylabel('Avg. expected return', size=15)
plt.xlabel('# of training steps (x100000)',size =14)  
plt.plot(avg_eco,color='purple',label='Eco-system')
#plt.plot(max_eco,color='blue',label='Eco-system')
plt.fill_between(x, avg_eco - eco_std_dev, avg_eco + eco_std_dev,facecolor="purple",color='purple',alpha=0.2)  

plt.plot(avg_ibac,color='orange',label='IBAC-SNI')
#plt.plot(max_ibac,color='red',label='IBAC-SNI')
plt.fill_between(x, avg_ibac - ibac_std_dev, avg_ibac + ibac_std_dev,facecolor="orange",color='orange',alpha=0.2)  
red_patch = mpatches.Patch(color='orange', label='IBAC-SNI')
blue_patch = mpatches.Patch(color='purple', label='eco-system')
plt.legend(handles=[red_patch,blue_patch])
plt.savefig('Benchmark_multiroom.pdf')
plt.close()
