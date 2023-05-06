import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
eco = []
eco.append(np.array([1,32,60,83,113,132,156,178,200,214,236]))
eco.append(np.array([1,31,54,72,95,119,135,158,174,195,212]))
eco.append(np.array([1,28,52,73,95,115,142,164,180,198,212]))
eco.append(np.array([1,26,52,81,108,131,151,168,192,212,229]))
eco.append(np.array([1,28,53,83,102,116,134,152,170,188,215]))
avg=0
avg_tsd =0
for i in range(5):
	avg=avg+eco[i]
avg_eco = avg/5
avg_eco_std = avg_eco
varx=[]
for i in range(5):
	vx = eco[i] - avg_eco
	vx = np.square(vx)
	varx.append(vx)
dev = 0
for i in range(5):
	dev = dev+varx[i]
dev = dev /5
eco_std_dev = np.sqrt(dev)
eco_std_error = eco_std_dev / math.sqrt(5)

eco = []
eco.append(np.array([1,28,47,68,93,120,145,167,188,206,227]))
eco.append(np.array([1,33,58,79,103,123,144,164,189,215,236]))
eco.append(np.array([1,26,47,73,99,118,139,164,183,207,223]))
eco.append(np.array([1,29,46,70,93,116,140,163,181,197,217]))
eco.append(np.array([1,36,61,82,99,120,143,160,176,196,218]))
avg=0
for i in range(5):
	avg=avg+eco[i]
avg_eco = avg/5
avg_eco_best = avg_eco
varx=[]
for i in range(5):
	vx = eco[i] - avg_eco
	vx = np.square(vx)
	varx.append(vx)
dev = 0
for i in range(5):
	dev = dev+varx[i]
dev = dev /5
eco_best_std_dev = np.sqrt(dev)
eco_best_std_error = eco_best_std_dev / math.sqrt(5)

eco = []
eco.append(np.array([1,35,54,73,95,112,137,155,173,192,208]))
eco.append(np.array([1,30,64,87,108,129,146,170,198,219,237]))
eco.append(np.array([1,31,63,89,118,143,162,181,195,211,229]))
eco.append(np.array([1,31,51,80,103,125,152,177,203,231,259]))
eco.append(np.array([1,24,54,82,104,126,158,177,192,210,227]))
avg=0
for i in range(5):
	avg=avg+eco[i]
avg_eco = avg/5
avg_eco_one = avg_eco
varx=[]
for i in range(5):
	vx = eco[i] - avg_eco
	vx = np.square(vx)
	varx.append(vx)
dev = 0
for i in range(5):
	dev = dev+varx[i]
dev = dev /5
eco_one_std_dev = np.sqrt(dev)
eco_one_std_error = eco_one_std_dev / math.sqrt(5)

eco = []
eco.append(np.array([1,34,58,86,113,143,172,197,222,249,274]))
eco.append(np.array([1,31,66,97,122,153,183,205,230,259,284]))
eco.append(np.array([1,36,69,90,112,142,170,195,225,248,272]))
eco.append(np.array([1,24,51,82,110,137,156,180,199,224,252]))
eco.append(np.array([1,33,67,95,122,142,169,193,216,244,271]))
avg=0
for i in range(5):
	avg=avg+eco[i]
avg_eco = avg/5
avg_eco_rnd = avg_eco
varx=[]
for i in range(5):
	vx = eco[i] - avg_eco
	vx = np.square(vx)
	varx.append(vx)
dev = 0
for i in range(5):
	dev = dev+varx[i]
dev = dev /5
eco_rnd_std_dev = np.sqrt(dev)
eco_rnd_std_error = eco_rnd_std_dev / math.sqrt(5)

x=[0,1,2,3,4,5,6,7,8,9,10]
plt.title("Number of agent in the pool (FourRoom)", size=15,y=1.06)
plt.xticks([0,2,4,6,8,10],[1,100,200,300,400,500],rotation=0)
plt.ylabel('# of agents', size=14)
plt.xlabel('# of environment trained',size =14)  
plt.plot(avg_eco_std,color='purple',label='Eco-system')
plt.plot(avg_eco_best,color='orange',label='Eco-system best')
plt.plot(avg_eco_one,color='green',label='Eco-system one')
plt.plot(avg_eco_rnd,color='blue',label='Eco-system random')
plt.fill_between(x, avg_eco_std - eco_std_error, avg_eco_std + eco_std_error,facecolor="purple",color='purple',alpha=0.2)  
plt.fill_between(x, avg_eco_best - eco_best_std_error, avg_eco_best + eco_best_std_error,facecolor="orange",color='orange',alpha=0.2)  
plt.fill_between(x, avg_eco_one - eco_one_std_error, avg_eco_one + eco_one_std_error,facecolor="green",color='green',alpha=0.2)
plt.fill_between(x, avg_eco_rnd - eco_rnd_std_error, avg_eco_rnd + eco_rnd_std_error,facecolor="blue",color='blue',alpha=0.2)  
purple_patch = mpatches.Patch(color='purple', label='no init.')
orange_patch = mpatches.Patch(color='orange', label='best init.')
green_patch = mpatches.Patch(color='green', label='forked agent init.')
blue_patch = mpatches.Patch(color='blue', label='random init.')
plt.legend(handles=[purple_patch,orange_patch,green_patch,blue_patch])
plt.savefig('Benchmark_fourroom_agents.pdf')
plt.close()
