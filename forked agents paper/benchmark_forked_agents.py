import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
eco = []
eco_tsd = []
for a in range(5):
	r_tmp = []
	r_tsd_tmp = []
	base_dir = "./Minigrid-standard/run"+str(a+1)+"/"
	for i in range(11):
		res_tab = np.load(base_dir+"Results_"+str(i*50)+".npy")
		res_avg = np.average(res_tab)
		calc_tab =[]
		for j in range(0,21):
			if res_tab[j]>0.8:	
				calc_tab.append(1)
			else:
				calc_tab.append(0)
		calc_tab=np.array(calc_tab)
		res_tsd = np.average(calc_tab)
		r_tmp.append(res_avg)
		r_tsd_tmp.append(res_tsd)
	eco.append(np.array(r_tmp))
	eco_tsd.append(np.array(r_tsd_tmp))	
avg=0
avg_tsd =0
for i in range(5):
	avg=avg+eco[i]
	avg_tsd = avg_tsd + eco_tsd[i]
avg_eco = avg/5
avg_eco_std = avg_eco
avg_eco_tsd = avg_tsd /5
avg_eco_std_tsd = avg_eco_tsd
varx=[]
varx_tsd=[]
for i in range(5):
	vx = eco[i] - avg_eco
	vx_tsd = eco_tsd[i] - avg_eco_tsd
	vx = np.square(vx)
	vx_tsd = np.square(vx_tsd)
	varx.append(vx)
	varx_tsd.append(vx_tsd)
dev = 0
dev_tsd = 0
for i in range(5):
	dev = dev+varx[i]
	dev_tsd = dev_tsd + varx_tsd[i]
dev = dev /5
dev_tsd = dev_tsd / 5
eco_std_dev = np.sqrt(dev)
eco_std_dev_tsd = np.sqrt(dev_tsd)
eco_std_error = eco_std_dev / math.sqrt(5)
eco_std_error_tsd = eco_std_dev_tsd / math.sqrt(5)

eco_best = []
eco_best_tsd = []
for a in range(5):
	r_tmp = []
	r_tsd_tmp = []
	base_dir = "./Minigrid-best/run"+str(a+1)+"/"
	for i in range(11):
		res_tab = np.load(base_dir+"Results_"+str(i*50)+".npy")
		res_avg = np.average(res_tab)
		calc_tab =[]
		for j in range(0,21):
			if res_tab[j]>0.8:	
				calc_tab.append(1)
			else:
				calc_tab.append(0)
		calc_tab=np.array(calc_tab)
		res_tsd = np.average(calc_tab)
		r_tmp.append(res_avg)
		r_tsd_tmp.append(res_tsd)
	eco_best.append(np.array(r_tmp))
	eco_best_tsd.append(np.array(r_tsd_tmp))	
eco = eco_best
eco_tsd = eco_best_tsd
avg=0
avg_tsd =0
for i in range(5):
	avg=avg+eco[i]
	avg_tsd = avg_tsd + eco_tsd[i]
avg_eco = avg/5
avg_eco_best = avg_eco
avg_eco_tsd = avg_tsd /5
avg_eco_best_tsd=avg_eco_tsd
varx=[]
varx_tsd=[]
for i in range(5):
	vx = eco[i] - avg_eco
	vx_tsd = eco_tsd[i] - avg_eco_tsd
	vx = np.square(vx)
	vx_tsd = np.square(vx_tsd)
	varx.append(vx)
	varx_tsd.append(vx_tsd)
dev = 0
dev_tsd = 0
for i in range(5):
	dev = dev+varx[i]
	dev_tsd = dev_tsd + varx_tsd[i]
dev = dev /5
dev_tsd = dev_tsd / 5
eco_best_std_dev = np.sqrt(dev)
eco_best_std_dev_tsd = np.sqrt(dev_tsd)
eco_best_std_error = eco_best_std_dev / math.sqrt(5)
eco_best_std_error_tsd = eco_best_std_dev_tsd / math.sqrt(5)

eco_one = []
eco_one_tsd = []
for a in range(5):
	r_tmp = []
	r_tsd_tmp = []
	base_dir = "./Minigrid-forked-agent/run"+str(a+1)+"/"
	for i in range(11):
		res_tab = np.load(base_dir+"Results_"+str(i*50)+".npy")
		res_avg = np.average(res_tab)
		calc_tab =[]
		for j in range(0,21):
			if res_tab[j]>0.8:	
				calc_tab.append(1)
			else:
				calc_tab.append(0)
		calc_tab=np.array(calc_tab)
		res_tsd = np.average(calc_tab)
		r_tmp.append(res_avg)
		r_tsd_tmp.append(res_tsd)
	eco_one.append(np.array(r_tmp))
	eco_one_tsd.append(np.array(r_tsd_tmp))	
eco = eco_one
eco_tsd = eco_one_tsd
avg=0
avg_tsd =0
for i in range(5):
	avg=avg+eco[i]
	avg_tsd = avg_tsd + eco_tsd[i]
avg_eco = avg/5
avg_eco_one = avg_eco
avg_eco_tsd = avg_tsd /5
avg_eco_one_tsd = avg_eco_tsd
varx=[]
varx_tsd=[]
for i in range(5):
	vx = eco[i] - avg_eco
	vx_tsd = eco_tsd[i] - avg_eco_tsd
	vx = np.square(vx)
	vx_tsd = np.square(vx_tsd)
	varx.append(vx)
	varx_tsd.append(vx_tsd)
dev = 0
dev_tsd = 0
for i in range(5):
	dev = dev+varx[i]
	dev_tsd = dev_tsd + varx_tsd[i]
dev = dev /5
dev_tsd = dev_tsd / 5
eco_one_std_dev = np.sqrt(dev)
eco_one_std_dev_tsd = np.sqrt(dev_tsd)
eco_one_std_error = eco_one_std_dev / math.sqrt(5)
eco_one_std_error_tsd = eco_one_std_dev_tsd / math.sqrt(5)

eco_rnd = []
eco_rnd_tsd = []
for a in range(5):
	r_tmp = []
	r_tsd_tmp = []
	base_dir = "./Minigrid-random/run"+str(a+1)+"/"
	for i in range(11):
		res_tab = np.load(base_dir+"Results_"+str(i*50)+".npy")
		res_avg = np.average(res_tab)
		calc_tab =[]
		for j in range(0,21):
			if res_tab[j]>0.8:	
				calc_tab.append(1)
			else:
				calc_tab.append(0)
		calc_tab=np.array(calc_tab)
		res_tsd = np.average(calc_tab)
		r_tmp.append(res_avg)
		r_tsd_tmp.append(res_tsd)
	eco_rnd.append(np.array(r_tmp))
	eco_rnd_tsd.append(np.array(r_tsd_tmp))	
eco = eco_rnd
eco_tsd = eco_rnd_tsd
avg=0
avg_tsd =0
for i in range(5):
	avg=avg+eco[i]
	avg_tsd = avg_tsd + eco_tsd[i]
avg_eco = avg/5
avg_eco_rnd = avg_eco
avg_eco_tsd = avg_tsd /5
avg_eco_rnd_tsd = avg_eco_tsd
varx=[]
varx_tsd=[]
for i in range(5):
	vx = eco[i] - avg_eco
	vx_tsd = eco_tsd[i] - avg_eco_tsd
	vx = np.square(vx)
	vx_tsd = np.square(vx_tsd)
	varx.append(vx)
	varx_tsd.append(vx_tsd)
dev = 0
dev_tsd = 0
for i in range(5):
	dev = dev+varx[i]
	dev_tsd = dev_tsd + varx_tsd[i]
dev = dev /5
dev_tsd = dev_tsd / 5
eco_rnd_std_dev = np.sqrt(dev)
eco_rnd_std_dev_tsd = np.sqrt(dev_tsd)
eco_rnd_std_error = eco_rnd_std_dev / math.sqrt(5)
eco_rnd_std_error_tsd = eco_rnd_std_dev_tsd / math.sqrt(5)

x=[0,1,2,3,4,5,6,7,8,9,10]
plt.title("Adaptability index on new environments (FourRoom)", size=15,y=1.06)
plt.xticks([0,2,4,6,8,10],[1,100,200,300,400,500],rotation=0)
plt.ylabel('Adaptability index based on avg. return', size=14)
plt.xlabel('# of environment trained',size =14)  
plt.plot(avg_eco_std,color='purple',label='Eco-system')
plt.plot(avg_eco_best,color='orange',label='Eco-system best')
plt.plot(avg_eco_one,color='green',label='Eco-system forked')
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
plt.savefig('Benchmark_fourroom_return.pdf')
plt.close()
