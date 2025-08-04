import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
ibac = np.array([0.05,0.09,0.21,0.20,0.24,0.23,0.23,0.22])
ibac2 = np.array([0.04,0.25,0.27,0.22,0.28,0.24,0.30,0.29])
ibac3 = np.array([0.01,0.04,0.13,0.05,0.06,0.03,0.07,0.05])
ibac4 = np.array([0.01,0.01,0.05,0.02,0.05,0.09,0.09,0.05])
ibac5 = np.array([0.06,0.20,0.22,0.24,0.30,0.29,0.31,0.31])

min_ibac = np.minimum(ibac,ibac2)
min_ibac = np.minimum(min_ibac,ibac3)
min_ibac = np.minimum(min_ibac,ibac4)
min_ibac = np.minimum(min_ibac,ibac5)

max_ibac = np.maximum(ibac,ibac2)
max_ibac = np.maximum(max_ibac,ibac3)
max_ibac = np.maximum(max_ibac,ibac4)
max_ibac = np.maximum(max_ibac,ibac5)

avg_ibac = ibac + ibac2+ibac3+ibac4+ibac5
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

eco = np.array([0.13,0.61,0.69,0.68,0.73,0.75,0.75,0.75])
eco2 = np.array([0.18,0.59,0.64,0.70,0.75,0.76,0.78,0.76])
eco3 = np.array([0.25,0.64,0.75,0.70,0.76,0.77,0.73,0.73])
eco4 = np.array([0.09,0.65,0.70,0.74,0.73,0.74,0.77,0.73])
eco5 = np.array([0.23,0.62,0.75,0.72,0.74,0.77,0.74,0.74])

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

reco = np.array([0.19,0.66,0.66,0.71,0.70,0.78,0.71,0.71])
reco2 = np.array([0.34,0.62,0.73,0.71,0.73,0.78,0.78,0.79])
reco3 = np.array([0.3,0.55,0.73,0.65,0.74,0.83,0.79,0.78])
reco4 = np.array([0.24,0.67,0.71,0.71,0.74,0.78,0.76,0.76])
reco5 = np.array([0.23,0.66,0.68,0.72,0.75,0.76,0.76,0.78])

min_reco = np.minimum(reco,reco2)
min_reco = np.minimum(min_reco,reco3)
min_reco = np.minimum(min_reco,reco4)
min_reco = np.minimum(min_reco,reco5)

max_reco = np.maximum(reco,reco2)
max_reco = np.maximum(max_reco,reco3)
max_reco = np.maximum(max_reco,reco4)
max_reco = np.maximum(max_reco,reco5)

avg_reco = reco+reco2+reco3+reco4+reco5
avg_reco = avg_reco /5

reco_var = reco - avg_reco
reco_var = np.square(reco_var)
reco_var2 = reco2 - avg_reco
reco_var2 = np.square(reco_var2)
reco_var3 = reco3 - avg_reco
reco_var3 = np.square(reco_var3)
reco_var4 = reco4 - avg_reco
reco_var4 = np.square(reco_var4)
reco_var5 = reco5 - avg_reco
reco_var5 = np.square(reco_var5)

reco_std_dev = reco_var + reco_var2 + reco_var3 + reco_var4 + reco_var5
reco_std_dev = reco_std_dev/5
reco_std_dev = np.sqrt(reco_std_dev)

ens = np.array([0.31,0.18,0.22,0.16,0.17,0.12,0.13,0.13])
ens2 = np.array([0.21,0.18,0.11,0.16,0.19,0.17,0.13,0.13])
ens3 = np.array([0.25,0.20,0.14,0.18,0.19,0.18,0.19,0.19])
ens4 = np.array([0.22,0.20,0.15,0.17,0.18,0.21,0.14,0.14])
ens5 = np.array([0.27,0.23,0.20,0.21,0.17,0.15,0.14,0.14])

min_ens = np.minimum(ens,ens2)
min_ens = np.minimum(min_ens,ens3)
min_ens = np.minimum(min_ens,ens4)
min_ens = np.minimum(min_ens,ens5)

max_ens = np.maximum(ens,ens2)
max_ens = np.maximum(max_ens,ens3)
max_ens = np.maximum(max_ens,ens4)
max_ens = np.maximum(max_ens,ens5)

avg_ens = ens+ens2+ens3+ens4+ens5
avg_ens = avg_ens /5

ens_var = ens - avg_ens
ens_var = np.square(ens_var)
ens_var2 = ens2 - avg_ens
ens_var2 = np.square(ens_var2)
ens_var3 = ens3 - avg_ens
ens_var3 = np.square(ens_var3)
ens_var4 = ens4 - avg_ens
ens_var4 = np.square(ens_var4)
ens_var5 = ens5 - avg_ens
ens_var5 = np.square(ens_var5)

ens_std_dev = ens_var + ens_var2 + ens_var3 + ens_var4 + ens_var5
ens_std_dev = ens_std_dev/5
ens_std_dev = np.sqrt(ens_std_dev)

x=[0,1,2,3,4,5,6,7]
plt.title("Generalizability index on new environments", size=16,y=1.06)
plt.xticks([0,1,2,3,4,5,6,7],[5,150,402,564,696,829,919,1000],rotation=0,size=15)
plt.yticks(size=15)
plt.ylabel('Avg. expected return', size=15)
plt.xlabel('# of training steps (x100000)',size =14)  

plt.plot(avg_reco,color='green',label='Eco-system reverse')
plt.fill_between(x, avg_reco - reco_std_dev, avg_reco + reco_std_dev,facecolor="green",color='green',alpha=0.2)  

plt.plot(avg_eco,color='purple',label='Eco-system')
plt.fill_between(x, avg_eco - eco_std_dev, avg_eco + eco_std_dev,facecolor="purple",color='purple',alpha=0.2)  

plt.plot(avg_ibac,color='orange',label='IBAC-SNI')
plt.fill_between(x, avg_ibac - ibac_std_dev, avg_ibac + ibac_std_dev,facecolor="orange",color='orange',alpha=0.2)  

plt.plot(avg_ens,color='blue',label='Ensemble voting')
plt.fill_between(x, avg_ens - ens_std_dev, avg_ens + ens_std_dev,facecolor="blue",color='blue',alpha=0.2)  

orange_patch = mpatches.Patch(color='orange', label='IBAC-SNI')
purple_patch = mpatches.Patch(color='purple', label='Eco-system')
green_patch = mpatches.Patch(color='green', label='Eco-system reverse')
lightblue_patch = mpatches.Patch(color='blue', label='Ensemble voting')
plt.legend(handles=[orange_patch,purple_patch,green_patch,lightblue_patch])
plt.savefig('Benchmark_fourroom.pdf')
plt.close()
