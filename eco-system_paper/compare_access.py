import numpy as np
np.random.seed(1231231)

from datetime import datetime
from scipy.special import softmax
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
import gym
from PIL import Image

import time

def generate_results_comparison():    
    plt.title("Number of accesses to environments", size=16,y=1.06)
    x = [1,2,3,4,5,6]
    plt.xticks([1,3,5,6],[50,150,250,300],rotation=0,size=15)
    plt.yticks(size=15)
    plt.ylabel('# accesses', size = 15)
    plt.xlabel('# of environment trained', size = 14)
    OAMT_training_access_0_easy = np.load('./Submarine-game-easy/One_agent_multi_training_access0.npy')
    MAOT_training_access_0_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_access0.npy')
    OAMT_training_access_1_easy = np.load('./Submarine-game-easy/One_agent_multi_training_access1.npy')
    MAOT_training_access_1_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_access1.npy')
    OAMT_training_access_2_easy = np.load('./Submarine-game-easy/One_agent_multi_training_access2.npy')
    MAOT_training_access_2_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_access2.npy')
    OAMT_training_access_3_easy = np.load('./Submarine-game-easy/One_agent_multi_training_access3.npy')
    MAOT_training_access_3_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_access3.npy')
    OAMT_training_access_4_easy = np.load('./Submarine-game-easy/One_agent_multi_training_access4.npy')
    MAOT_training_access_4_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_access4.npy')
    for i in range(7,21):
        OAMT_training_access_0_easy = np.delete(OAMT_training_access_0_easy,7)
        MAOT_training_access_0_easy = np.delete(MAOT_training_access_0_easy,7)
        OAMT_training_access_1_easy = np.delete(OAMT_training_access_1_easy,7)
        MAOT_training_access_1_easy = np.delete(MAOT_training_access_1_easy,7)
        OAMT_training_access_2_easy = np.delete(OAMT_training_access_2_easy,7)
        MAOT_training_access_2_easy = np.delete(MAOT_training_access_2_easy,7)
        OAMT_training_access_3_easy = np.delete(OAMT_training_access_3_easy,7)
        MAOT_training_access_3_easy = np.delete(MAOT_training_access_3_easy,7)
        OAMT_training_access_4_easy = np.delete(OAMT_training_access_4_easy,7)
        MAOT_training_access_4_easy = np.delete(MAOT_training_access_4_easy,7)

    OAMT_training_access_0_easy = np.delete(OAMT_training_access_0_easy,0)
    MAOT_training_access_0_easy = np.delete(MAOT_training_access_0_easy,0)
    OAMT_training_access_1_easy = np.delete(OAMT_training_access_1_easy,0)
    MAOT_training_access_1_easy = np.delete(MAOT_training_access_1_easy,0)
    OAMT_training_access_2_easy = np.delete(OAMT_training_access_2_easy,0)
    MAOT_training_access_2_easy = np.delete(MAOT_training_access_2_easy,0)
    OAMT_training_access_3_easy = np.delete(OAMT_training_access_3_easy,0)
    MAOT_training_access_3_easy = np.delete(MAOT_training_access_3_easy,0)
    OAMT_training_access_4_easy = np.delete(OAMT_training_access_4_easy,0)
    MAOT_training_access_4_easy = np.delete(MAOT_training_access_4_easy,0)
    
    
    min_OAMT_easy = np.minimum(OAMT_training_access_0_easy,OAMT_training_access_1_easy)
    min_OAMT_easy = np.minimum(min_OAMT_easy,OAMT_training_access_2_easy)
    min_OAMT_easy = np.minimum(min_OAMT_easy,OAMT_training_access_3_easy)
    min_OAMT_easy = np.minimum(min_OAMT_easy,OAMT_training_access_4_easy)
    
    max_OAMT_easy = np.maximum(OAMT_training_access_0_easy,OAMT_training_access_1_easy)
    max_OAMT_easy = np.maximum(max_OAMT_easy,OAMT_training_access_2_easy)
    max_OAMT_easy = np.maximum(max_OAMT_easy,OAMT_training_access_3_easy)
    max_OAMT_easy = np.maximum(max_OAMT_easy,OAMT_training_access_4_easy) 
    
    avg_OAMT_easy = OAMT_training_access_0_easy + OAMT_training_access_1_easy + OAMT_training_access_2_easy + OAMT_training_access_3_easy + OAMT_training_access_4_easy
    avg_OAMT_easy = avg_OAMT_easy / 5 
        
    OAMT_var_easy = OAMT_training_access_0_easy - avg_OAMT_easy
    OAMT_var_easy = np.square(OAMT_var_easy)
    OAMT_var2_easy = OAMT_training_access_1_easy - avg_OAMT_easy
    OAMT_var2_easy = np.square(OAMT_var2_easy)
    OAMT_var3_easy = OAMT_training_access_2_easy - avg_OAMT_easy
    OAMT_var3_easy = np.square(OAMT_var3_easy)
    OAMT_var4_easy = OAMT_training_access_3_easy - avg_OAMT_easy
    OAMT_var4_easy = np.square(OAMT_var4_easy)
    OAMT_var5_easy = OAMT_training_access_4_easy - avg_OAMT_easy
    OAMT_var5_easy = np.square(OAMT_var5_easy)
    
    OAMT_std_dev_easy = OAMT_var_easy + OAMT_var2_easy + OAMT_var3_easy + OAMT_var4_easy + OAMT_var5_easy
    OAMT_std_dev_easy = OAMT_std_dev_easy/5
    OAMT_std_dev_easy = np.sqrt(OAMT_std_dev_easy)
    
    min_MAOT_easy = np.minimum(MAOT_training_access_0_easy,MAOT_training_access_1_easy)
    min_MAOT_easy = np.minimum(min_MAOT_easy,MAOT_training_access_2_easy)
    min_MAOT_easy = np.minimum(min_MAOT_easy,MAOT_training_access_3_easy)
    min_MAOT_easy = np.minimum(min_MAOT_easy,MAOT_training_access_4_easy)
    
    max_MAOT_easy = np.maximum(MAOT_training_access_0_easy,MAOT_training_access_1_easy)
    max_MAOT_easy = np.maximum(max_MAOT_easy,MAOT_training_access_2_easy)
    max_MAOT_easy = np.maximum(max_MAOT_easy,MAOT_training_access_3_easy)
    max_MAOT_easy = np.maximum(max_MAOT_easy,MAOT_training_access_4_easy)       
    
    avg_MAOT_easy = MAOT_training_access_0_easy + MAOT_training_access_1_easy + MAOT_training_access_2_easy + MAOT_training_access_3_easy + MAOT_training_access_4_easy
    avg_MAOT_easy = avg_MAOT_easy / 5
    
    MAOT_var_easy = MAOT_training_access_0_easy - avg_MAOT_easy
    MAOT_var_easy = np.square(MAOT_var_easy)
    MAOT_var2_easy = MAOT_training_access_1_easy - avg_MAOT_easy
    MAOT_var2_easy = np.square(MAOT_var2_easy)
    MAOT_var3_easy = MAOT_training_access_2_easy - avg_MAOT_easy
    MAOT_var3_easy = np.square(MAOT_var3_easy)
    MAOT_var4_easy = MAOT_training_access_3_easy - avg_MAOT_easy
    MAOT_var4_easy = np.square(MAOT_var4_easy)
    MAOT_var5_easy = MAOT_training_access_4_easy - avg_MAOT_easy
    MAOT_var5_easy = np.square(MAOT_var5_easy)
    
    MAOT_std_dev_easy = MAOT_var_easy + MAOT_var2_easy + MAOT_var3_easy + MAOT_var4_easy + MAOT_var5_easy
    MAOT_std_dev_easy = MAOT_std_dev_easy/5
    MAOT_std_dev_easy = np.sqrt(MAOT_std_dev_easy)
    
    OAMT_training_access_0 = np.load('./Submarine-game-hard/One_agent_multi_training_access0.npy')
    MAOT_training_access_0 = np.load('./Submarine-game-hard/Multi_agent_one_training_access0.npy')
    OAMT_training_access_1 = np.load('./Submarine-game-hard/One_agent_multi_training_access1.npy')
    MAOT_training_access_1 = np.load('./Submarine-game-hard/Multi_agent_one_training_access1.npy')
    OAMT_training_access_2 = np.load('./Submarine-game-hard/One_agent_multi_training_access2.npy')
    MAOT_training_access_2 = np.load('./Submarine-game-hard/Multi_agent_one_training_access2.npy')
    OAMT_training_access_3 = np.load('./Submarine-game-hard/One_agent_multi_training_access3.npy')
    MAOT_training_access_3 = np.load('./Submarine-game-hard/Multi_agent_one_training_access3.npy')
    OAMT_training_access_4 = np.load('./Submarine-game-hard/One_agent_multi_training_access4.npy')
    MAOT_training_access_4 = np.load('./Submarine-game-hard/Multi_agent_one_training_access4.npy')
    OAMT_training_access_0 = np.delete(OAMT_training_access_0,0)
    MAOT_training_access_0 = np.delete(MAOT_training_access_0,0)
    OAMT_training_access_1 = np.delete(OAMT_training_access_1,0)
    MAOT_training_access_1 = np.delete(MAOT_training_access_1,0)
    OAMT_training_access_2 = np.delete(OAMT_training_access_2,0)
    MAOT_training_access_2 = np.delete(MAOT_training_access_2,0)
    OAMT_training_access_3 = np.delete(OAMT_training_access_3,0)
    MAOT_training_access_3 = np.delete(MAOT_training_access_3,0)
    OAMT_training_access_4 = np.delete(OAMT_training_access_4,0)
    MAOT_training_access_4 = np.delete(MAOT_training_access_4,0)
    
    min_OAMT = np.minimum(OAMT_training_access_0,OAMT_training_access_1)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_access_2)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_access_3)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_access_4)
    
    max_OAMT = np.maximum(OAMT_training_access_0,OAMT_training_access_1)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_access_2)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_access_3)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_access_4) 
    
    avg_OAMT = OAMT_training_access_0 + OAMT_training_access_1+OAMT_training_access_2+OAMT_training_access_3+OAMT_training_access_4
    avg_OAMT = avg_OAMT / 5 
        
    OAMT_var = OAMT_training_access_0 - avg_OAMT
    OAMT_var = np.square(OAMT_var)
    OAMT_var2 = OAMT_training_access_1 - avg_OAMT
    OAMT_var2 = np.square(OAMT_var2)
    OAMT_var3 = OAMT_training_access_2 - avg_OAMT
    OAMT_var3 = np.square(OAMT_var3)
    OAMT_var4 = OAMT_training_access_3 - avg_OAMT
    OAMT_var4 = np.square(OAMT_var4)
    OAMT_var5 = OAMT_training_access_4 - avg_OAMT
    OAMT_var5 = np.square(OAMT_var5)
    
    OAMT_std_dev = OAMT_var + OAMT_var2 + OAMT_var3 + OAMT_var4 + OAMT_var5
    OAMT_std_dev = OAMT_std_dev/5
    OAMT_std_dev = np.sqrt(OAMT_std_dev)
    
    min_MAOT = np.minimum(MAOT_training_access_0,MAOT_training_access_1)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_access_2)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_access_3)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_access_4)
    
    max_MAOT = np.maximum(MAOT_training_access_0,MAOT_training_access_1)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_access_2)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_access_3)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_access_4)       
    
    avg_MAOT = MAOT_training_access_0 + MAOT_training_access_1+MAOT_training_access_2+MAOT_training_access_3+MAOT_training_access_4
    avg_MAOT = avg_MAOT / 5
    
    MAOT_var = MAOT_training_access_0 - avg_MAOT
    MAOT_var = np.square(MAOT_var)
    MAOT_var2 = MAOT_training_access_1 - avg_MAOT
    MAOT_var2 = np.square(MAOT_var2)
    MAOT_var3 = MAOT_training_access_2 - avg_MAOT
    MAOT_var3 = np.square(MAOT_var3)
    MAOT_var4 = MAOT_training_access_3 - avg_MAOT
    MAOT_var4 = np.square(MAOT_var4)
    MAOT_var5 = MAOT_training_access_4 - avg_MAOT
    MAOT_var5 = np.square(MAOT_var5)
    
    MAOT_std_dev = MAOT_var + MAOT_var2 + MAOT_var3 + MAOT_var4 + MAOT_var5
    MAOT_std_dev = MAOT_std_dev/5
    MAOT_std_dev = np.sqrt(MAOT_std_dev)
    
     
    plt.plot(x,avg_MAOT_easy, color='purple', label='eco-system access (easy)')
    plt.fill_between(x,avg_MAOT_easy-MAOT_std_dev_easy, avg_MAOT_easy+MAOT_std_dev_easy,facecolor="purple", color='purple',alpha=0.2)  
    plt.plot(x,avg_OAMT_easy, color='orange', label='one-agent access (easy)')
    plt.fill_between(x,avg_OAMT_easy-OAMT_std_dev_easy, avg_OAMT_easy+OAMT_std_dev_easy,facecolor="orange", color='orange',alpha=0.2)  
    orange_patch = mpatches.Patch(color='orange', label='single-agent (submarine easy)')
    purple_patch = mpatches.Patch(color='purple', label='eco-system (submarie easy)')
    plt.plot(x,avg_MAOT, color='blue', label='eco-system access')
    plt.fill_between(x,avg_MAOT-MAOT_std_dev, avg_MAOT+MAOT_std_dev,facecolor="blue", color='blue',alpha=0.2)  
    plt.plot(x,avg_OAMT, color='green', label='one-agent access')
    plt.fill_between(x,avg_OAMT-OAMT_std_dev, avg_OAMT+OAMT_std_dev,facecolor="green", color='green',alpha=0.2)  
    green_patch = mpatches.Patch(color='green', label='single-agent (submarine hard)')
    blue_patch = mpatches.Patch(color='blue', label='eco-system (submarine hard)')
    plt.legend(handles=[orange_patch,purple_patch,green_patch,blue_patch])
    plt.savefig('Architecture_comparison_access.pdf')
    plt.close()
    

def main():
    generate_results_comparison()

if __name__ == "__main__":
    main()
