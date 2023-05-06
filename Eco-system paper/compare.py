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
    plt.title("Training time on initial environments", size=16,y=1.06)
    x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    plt.xticks([1,4,8,12,16,20],[50,200,400,600,800,1000],rotation=0,size=15)
    plt.yticks(size=15)
    plt.ylabel('seconds', size=15)
    plt.xlabel('# of environment trained' , size=14)
    OAMT_training_time_0_easy = np.load('./Submarine-game-easy/One_agent_multi_training_time0.npy')
    MAOT_training_time_0_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_time0.npy')
    OAMT_training_time_1_easy = np.load('./Submarine-game-easy/One_agent_multi_training_time1.npy')
    MAOT_training_time_1_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_time1.npy')
    OAMT_training_time_2_easy = np.load('./Submarine-game-easy/One_agent_multi_training_time2.npy')
    MAOT_training_time_2_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_time2.npy')
    OAMT_training_time_3_easy = np.load('./Submarine-game-easy/One_agent_multi_training_time3.npy')
    MAOT_training_time_3_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_time3.npy')
    OAMT_training_time_4_easy = np.load('./Submarine-game-easy/One_agent_multi_training_time4.npy')
    MAOT_training_time_4_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_time4.npy')
    OAMT_training_time_0_easy = np.delete(OAMT_training_time_0_easy,0)
    MAOT_training_time_0_easy = np.delete(MAOT_training_time_0_easy,0)
    OAMT_training_time_1_easy = np.delete(OAMT_training_time_1_easy,0)
    MAOT_training_time_1_easy = np.delete(MAOT_training_time_1_easy,0)
    OAMT_training_time_2_easy = np.delete(OAMT_training_time_2_easy,0)
    MAOT_training_time_2_easy = np.delete(MAOT_training_time_2_easy,0)
    OAMT_training_time_3_easy = np.delete(OAMT_training_time_3_easy,0)
    MAOT_training_time_3_easy = np.delete(MAOT_training_time_3_easy,0)
    OAMT_training_time_4_easy = np.delete(OAMT_training_time_4_easy,0)
    MAOT_training_time_4_easy = np.delete(MAOT_training_time_4_easy,0)

    min_OAMT_easy = np.minimum(OAMT_training_time_0_easy,OAMT_training_time_1_easy)
    min_OAMT_easy = np.minimum(min_OAMT_easy,OAMT_training_time_2_easy)
    min_OAMT_easy = np.minimum(min_OAMT_easy,OAMT_training_time_3_easy)
    min_OAMT_easy = np.minimum(min_OAMT_easy,OAMT_training_time_4_easy)
    
    max_OAMT_easy = np.maximum(OAMT_training_time_0_easy,OAMT_training_time_1_easy)
    max_OAMT_easy = np.maximum(max_OAMT_easy,OAMT_training_time_2_easy)
    max_OAMT_easy = np.maximum(max_OAMT_easy,OAMT_training_time_3_easy)
    max_OAMT_easy = np.maximum(max_OAMT_easy,OAMT_training_time_4_easy)
    
    avg_OAMT_easy = OAMT_training_time_0_easy + OAMT_training_time_1_easy + OAMT_training_time_2_easy + OAMT_training_time_3_easy + OAMT_training_time_4_easy
    avg_OAMT_easy = avg_OAMT_easy / 5
    
    OAMT_var_easy = OAMT_training_time_0_easy - avg_OAMT_easy
    OAMT_var_easy = np.square(OAMT_var_easy)
    OAMT_var2_easy = OAMT_training_time_1_easy - avg_OAMT_easy
    OAMT_var2_easy = np.square(OAMT_var2_easy)
    OAMT_var3_easy = OAMT_training_time_2_easy - avg_OAMT_easy
    OAMT_var3_easy = np.square(OAMT_var3_easy)
    OAMT_var4_easy = OAMT_training_time_3_easy - avg_OAMT_easy
    OAMT_var4_easy = np.square(OAMT_var4_easy)
    OAMT_var5_easy = OAMT_training_time_4_easy - avg_OAMT_easy
    OAMT_var5_easy = np.square(OAMT_var5_easy)
    
    OAMT_std_dev_easy = OAMT_var_easy + OAMT_var2_easy + OAMT_var3_easy + OAMT_var4_easy + OAMT_var5_easy
    OAMT_std_dev_easy = OAMT_std_dev_easy/5
    OAMT_std_dev_easy = np.sqrt(OAMT_std_dev_easy)
    
    min_MAOT_easy = np.minimum(MAOT_training_time_0_easy,MAOT_training_time_1_easy)
    min_MAOT_easy = np.minimum(min_MAOT_easy,MAOT_training_time_2_easy)
    min_MAOT_easy = np.minimum(min_MAOT_easy,MAOT_training_time_3_easy)
    min_MAOT_easy = np.minimum(min_MAOT_easy,MAOT_training_time_4_easy)
    
    max_MAOT_easy = np.maximum(MAOT_training_time_0_easy,MAOT_training_time_1_easy)
    max_MAOT_easy = np.maximum(max_MAOT_easy,MAOT_training_time_2_easy)
    max_MAOT_easy = np.maximum(max_MAOT_easy,MAOT_training_time_3_easy)
    max_MAOT_easy = np.maximum(max_MAOT_easy,MAOT_training_time_4_easy)        

    avg_MAOT_easy = MAOT_training_time_0_easy + MAOT_training_time_1_easy + MAOT_training_time_2_easy + MAOT_training_time_3_easy + MAOT_training_time_4_easy
    avg_MAOT_easy = avg_MAOT_easy / 5

    MAOT_var_easy = MAOT_training_time_0_easy - avg_MAOT_easy
    MAOT_var_easy = np.square(MAOT_var_easy)
    MAOT_var2_easy = MAOT_training_time_1_easy - avg_MAOT_easy
    MAOT_var2_easy = np.square(MAOT_var2_easy)
    MAOT_var3_easy = MAOT_training_time_2_easy - avg_MAOT_easy
    MAOT_var3_easy = np.square(MAOT_var3_easy)
    MAOT_var4_easy = MAOT_training_time_3_easy - avg_MAOT_easy
    MAOT_var4_easy = np.square(MAOT_var4_easy)
    MAOT_var5_easy = MAOT_training_time_4_easy - avg_MAOT_easy
    MAOT_var5_easy = np.square(MAOT_var5_easy)

    MAOT_std_dev_easy = MAOT_var_easy + MAOT_var2_easy + MAOT_var3_easy + MAOT_var4_easy + MAOT_var5_easy
    MAOT_std_dev_easy = MAOT_std_dev_easy/5
    MAOT_std_dev_easy = np.sqrt(MAOT_std_dev_easy)
    
    OAMT_training_time_0 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_time0.npy')
    MAOT_training_time_0 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_time0.npy')
    OAMT_training_time_1 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_time1.npy')
    MAOT_training_time_1 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_time1.npy')
    OAMT_training_time_2 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_time2.npy')
    MAOT_training_time_2 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_time2.npy')
    OAMT_training_time_3 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_time3.npy')
    MAOT_training_time_3 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_time3.npy')
    OAMT_training_time_4 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_time4.npy')
    MAOT_training_time_4 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_time4.npy')
    OAMT_training_time_0 = np.delete(OAMT_training_time_0,0)
    MAOT_training_time_0 = np.delete(MAOT_training_time_0,0)
    OAMT_training_time_1 = np.delete(OAMT_training_time_1,0)
    MAOT_training_time_1 = np.delete(MAOT_training_time_1,0)
    OAMT_training_time_2 = np.delete(OAMT_training_time_2,0)
    MAOT_training_time_2 = np.delete(MAOT_training_time_2,0)
    OAMT_training_time_3 = np.delete(OAMT_training_time_3,0)
    MAOT_training_time_3 = np.delete(MAOT_training_time_3,0)
    OAMT_training_time_4 = np.delete(OAMT_training_time_4,0)
    MAOT_training_time_4 = np.delete(MAOT_training_time_4,0)
    
    min_OAMT = np.minimum(OAMT_training_time_0,OAMT_training_time_1)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_time_2)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_time_3)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_time_4)
    
    max_OAMT = np.maximum(OAMT_training_time_0,OAMT_training_time_1)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_time_2)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_time_3)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_time_4)
    
    avg_OAMT = OAMT_training_time_0 + OAMT_training_time_1+OAMT_training_time_2+OAMT_training_time_3+OAMT_training_time_4
    avg_OAMT = avg_OAMT / 5
    
    OAMT_var = OAMT_training_time_0 - avg_OAMT
    OAMT_var = np.square(OAMT_var)
    OAMT_var2 = OAMT_training_time_1 - avg_OAMT
    OAMT_var2 = np.square(OAMT_var2)
    OAMT_var3 = OAMT_training_time_2 - avg_OAMT
    OAMT_var3 = np.square(OAMT_var3)
    OAMT_var4 = OAMT_training_time_3 - avg_OAMT
    OAMT_var4 = np.square(OAMT_var4)
    OAMT_var5 = OAMT_training_time_4 - avg_OAMT
    OAMT_var5 = np.square(OAMT_var5)
    
    OAMT_std_dev = OAMT_var + OAMT_var2 + OAMT_var3 + OAMT_var4 + OAMT_var5
    OAMT_std_dev = OAMT_std_dev/5
    OAMT_std_dev = np.sqrt(OAMT_std_dev)
    
    min_MAOT = np.minimum(MAOT_training_time_0,MAOT_training_time_1)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_time_2)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_time_3)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_time_4)
    
    max_MAOT = np.maximum(MAOT_training_time_0,MAOT_training_time_1)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_time_2)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_time_3)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_time_4)        
    
    avg_MAOT = MAOT_training_time_0 + MAOT_training_time_1+MAOT_training_time_2+MAOT_training_time_3+MAOT_training_time_4
    avg_MAOT = avg_MAOT / 5
    
    MAOT_var = MAOT_training_time_0 - avg_MAOT
    MAOT_var = np.square(MAOT_var)
    MAOT_var2 = MAOT_training_time_1 - avg_MAOT
    MAOT_var2 = np.square(MAOT_var2)
    MAOT_var3 = MAOT_training_time_2 - avg_MAOT
    MAOT_var3 = np.square(MAOT_var3)
    MAOT_var4 = MAOT_training_time_3 - avg_MAOT
    MAOT_var4 = np.square(MAOT_var4)
    MAOT_var5 = MAOT_training_time_4 - avg_MAOT
    MAOT_var5 = np.square(MAOT_var5)
    
    MAOT_std_dev = MAOT_var + MAOT_var2 + MAOT_var3 + MAOT_var4 + MAOT_var5
    MAOT_std_dev = MAOT_std_dev/5
    MAOT_std_dev = np.sqrt(MAOT_std_dev)


    plt.plot(x,avg_MAOT_easy, color='purple', label='eco-system duration DDQN')
    plt.fill_between(x,avg_MAOT_easy-MAOT_std_dev_easy, avg_MAOT_easy+MAOT_std_dev_easy,facecolor="purple", color='purple',alpha=0.2)  
    plt.plot(x,avg_OAMT_easy, color='orange', label='one-agent duration DDQN')
    plt.fill_between(x,avg_OAMT_easy-OAMT_std_dev_easy, avg_OAMT_easy+OAMT_std_dev_easy,facecolor="orange", color='orange',alpha=0.2)  
    orange_patch = mpatches.Patch(color='orange', label='single-agent DDQN')
    purple_patch = mpatches.Patch(color='purple', label='eco-system DDQN')
    plt.legend(handles=[orange_patch,purple_patch])
    plt.savefig('Architecture_comparison_duration_easy.pdf')
    plt.close()

    plt.title("Catastrophic forgetting avoidance index", size=16,y=1.06)
    x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    plt.xticks([1,4,8,12,16,20],[50, 200,400,600,800,1000],rotation=0,size=15)
    plt.yticks(size=15)
    plt.ylabel('% accuracy', size=15)
    plt.xlabel('# of environment trained', size=14)
    OAMT_training_forget_0_easy = np.load('./Submarine-game-easy/One_agent_multi_training_forget0.npy')
    MAOT_training_forget_0_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_forget0.npy')
    OAMT_training_forget_1_easy = np.load('./Submarine-game-easy/One_agent_multi_training_forget1.npy')
    MAOT_training_forget_1_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_forget1.npy')
    OAMT_training_forget_2_easy = np.load('./Submarine-game-easy/One_agent_multi_training_forget2.npy')
    MAOT_training_forget_2_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_forget2.npy')
    OAMT_training_forget_3_easy = np.load('./Submarine-game-easy/One_agent_multi_training_forget3.npy')
    MAOT_training_forget_3_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_forget3.npy')
    OAMT_training_forget_4_easy = np.load('./Submarine-game-easy/One_agent_multi_training_forget4.npy')
    MAOT_training_forget_4_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_forget4.npy')
    OAMT_training_forget_0_easy = np.delete(OAMT_training_forget_0_easy,0)
    MAOT_training_forget_0_easy = np.delete(MAOT_training_forget_0_easy,0)
    OAMT_training_forget_1_easy = np.delete(OAMT_training_forget_1_easy,0)
    MAOT_training_forget_1_easy = np.delete(MAOT_training_forget_1_easy,0)
    OAMT_training_forget_2_easy = np.delete(OAMT_training_forget_2_easy,0)
    MAOT_training_forget_2_easy = np.delete(MAOT_training_forget_2_easy,0)
    OAMT_training_forget_3_easy = np.delete(OAMT_training_forget_3_easy,0)
    MAOT_training_forget_3_easy = np.delete(MAOT_training_forget_3_easy,0)
    OAMT_training_forget_4_easy = np.delete(OAMT_training_forget_4_easy,0)
    MAOT_training_forget_4_easy = np.delete(MAOT_training_forget_4_easy,0)
    
    min_OAMT_easy = np.minimum(OAMT_training_forget_0_easy,OAMT_training_forget_1_easy)
    min_OAMT_easy = np.minimum(min_OAMT_easy,OAMT_training_forget_2_easy)
    min_OAMT_easy = np.minimum(min_OAMT_easy,OAMT_training_forget_3_easy)
    min_OAMT_easy = np.minimum(min_OAMT_easy,OAMT_training_forget_4_easy)

    max_OAMT_easy = np.maximum(OAMT_training_forget_0_easy,OAMT_training_forget_1_easy)
    max_OAMT_easy = np.maximum(max_OAMT_easy,OAMT_training_forget_2_easy)
    max_OAMT_easy = np.maximum(max_OAMT_easy,OAMT_training_forget_3_easy)
    max_OAMT_easy = np.maximum(max_OAMT_easy,OAMT_training_forget_4_easy)
    
    avg_OAMT_easy = OAMT_training_forget_0_easy + OAMT_training_forget_1_easy + OAMT_training_forget_2_easy + OAMT_training_forget_3_easy + OAMT_training_forget_4_easy
    avg_OAMT_easy = avg_OAMT_easy / 5 
    
        
    OAMT_var_easy = OAMT_training_forget_0_easy - avg_OAMT_easy
    OAMT_var_easy = np.square(OAMT_var_easy)
    OAMT_var2_easy = OAMT_training_forget_1_easy - avg_OAMT_easy
    OAMT_var2_easy = np.square(OAMT_var2_easy)
    OAMT_var3_easy = OAMT_training_forget_2_easy - avg_OAMT_easy
    OAMT_var3_easy = np.square(OAMT_var3_easy)
    OAMT_var4_easy = OAMT_training_forget_3_easy - avg_OAMT_easy
    OAMT_var4_easy = np.square(OAMT_var4_easy)
    OAMT_var5_easy = OAMT_training_forget_4_easy - avg_OAMT_easy
    OAMT_var5_easy = np.square(OAMT_var5_easy)
    
    OAMT_std_dev_easy = OAMT_var_easy + OAMT_var2_easy + OAMT_var3_easy + OAMT_var4_easy + OAMT_var5_easy
    OAMT_std_dev_easy = OAMT_std_dev_easy/5
    OAMT_std_dev_easy = np.sqrt(OAMT_std_dev_easy)

    min_MAOT_easy = np.minimum(MAOT_training_forget_0_easy,MAOT_training_forget_1_easy)
    min_MAOT_easy = np.minimum(min_MAOT_easy,MAOT_training_forget_2_easy)
    min_MAOT_easy = np.minimum(min_MAOT_easy,MAOT_training_forget_3_easy)
    min_MAOT_easy = np.minimum(min_MAOT_easy,MAOT_training_forget_4_easy)

    max_MAOT_easy = np.maximum(MAOT_training_forget_0_easy,MAOT_training_forget_1_easy)
    max_MAOT_easy = np.maximum(max_MAOT_easy,MAOT_training_forget_2_easy)
    max_MAOT_easy = np.maximum(max_MAOT_easy,MAOT_training_forget_3_easy)
    max_MAOT_easy = np.maximum(max_MAOT_easy,MAOT_training_forget_4_easy)       
    
    avg_MAOT_easy = MAOT_training_forget_0_easy + MAOT_training_forget_1_easy + MAOT_training_forget_2_easy + MAOT_training_forget_3_easy + MAOT_training_forget_4_easy
    avg_MAOT_easy = avg_MAOT_easy / 5
    
    MAOT_var_easy = MAOT_training_forget_0_easy - avg_MAOT_easy
    MAOT_var_easy = np.square(MAOT_var_easy)
    MAOT_var2_easy = MAOT_training_forget_1_easy - avg_MAOT_easy
    MAOT_var2_easy = np.square(MAOT_var2_easy)
    MAOT_var3_easy = MAOT_training_forget_2_easy - avg_MAOT_easy
    MAOT_var3_easy = np.square(MAOT_var3_easy)
    MAOT_var4_easy = MAOT_training_forget_3_easy - avg_MAOT_easy
    MAOT_var4_easy = np.square(MAOT_var4_easy)
    MAOT_var5_easy = MAOT_training_forget_4_easy - avg_MAOT_easy
    MAOT_var5_easy = np.square(MAOT_var5_easy)
    
    MAOT_std_dev_easy = MAOT_var_easy + MAOT_var2_easy + MAOT_var3_easy + MAOT_var4_easy + MAOT_var5_easy
    MAOT_std_dev_easy = MAOT_std_dev_easy/5
    MAOT_std_dev_easy = np.sqrt(MAOT_std_dev_easy)
    
    OAMT_training_forget_0 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_forget0.npy')
    MAOT_training_forget_0 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_forget0.npy')
    OAMT_training_forget_1 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_forget1.npy')
    MAOT_training_forget_1 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_forget1.npy')
    OAMT_training_forget_2 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_forget2.npy')
    MAOT_training_forget_2 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_forget2.npy')
    OAMT_training_forget_3 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_forget3.npy')
    MAOT_training_forget_3 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_forget3.npy')
    OAMT_training_forget_4 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_forget4.npy')
    MAOT_training_forget_4 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_forget4.npy')
    OAMT_training_forget_0 = np.delete(OAMT_training_forget_0,0)
    MAOT_training_forget_0 = np.delete(MAOT_training_forget_0,0)
    OAMT_training_forget_1 = np.delete(OAMT_training_forget_1,0)
    MAOT_training_forget_1 = np.delete(MAOT_training_forget_1,0)
    OAMT_training_forget_2 = np.delete(OAMT_training_forget_2,0)
    MAOT_training_forget_2 = np.delete(MAOT_training_forget_2,0)
    OAMT_training_forget_3 = np.delete(OAMT_training_forget_3,0)
    MAOT_training_forget_3 = np.delete(MAOT_training_forget_3,0)
    OAMT_training_forget_4 = np.delete(OAMT_training_forget_4,0)
    MAOT_training_forget_4 = np.delete(MAOT_training_forget_4,0)
    
    min_OAMT = np.minimum(OAMT_training_forget_0,OAMT_training_forget_1)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_forget_2)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_forget_3)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_forget_4)
    
    max_OAMT = np.maximum(OAMT_training_forget_0,OAMT_training_forget_1)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_forget_2)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_forget_3)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_forget_4)
    
    avg_OAMT = OAMT_training_forget_0 + OAMT_training_forget_1+OAMT_training_forget_2+OAMT_training_forget_3+OAMT_training_forget_4
    avg_OAMT = avg_OAMT / 5 
    
        
    OAMT_var = OAMT_training_forget_0 - avg_OAMT
    OAMT_var = np.square(OAMT_var)
    OAMT_var2 = OAMT_training_forget_1 - avg_OAMT
    OAMT_var2 = np.square(OAMT_var2)
    OAMT_var3 = OAMT_training_forget_2 - avg_OAMT
    OAMT_var3 = np.square(OAMT_var3)
    OAMT_var4 = OAMT_training_forget_3 - avg_OAMT
    OAMT_var4 = np.square(OAMT_var4)
    OAMT_var5 = OAMT_training_forget_4 - avg_OAMT
    OAMT_var5 = np.square(OAMT_var5)
    
    OAMT_std_dev = OAMT_var + OAMT_var2 + OAMT_var3 + OAMT_var4 + OAMT_var5
    OAMT_std_dev = OAMT_std_dev/5
    OAMT_std_dev = np.sqrt(OAMT_std_dev)
    
    min_MAOT = np.minimum(MAOT_training_forget_0,MAOT_training_forget_1)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_forget_2)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_forget_3)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_forget_4)
    
    max_MAOT = np.maximum(MAOT_training_forget_0,MAOT_training_forget_1)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_forget_2)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_forget_3)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_forget_4)       
    
    avg_MAOT = MAOT_training_forget_0 + MAOT_training_forget_1+MAOT_training_forget_2+MAOT_training_forget_3+MAOT_training_forget_4
    avg_MAOT = avg_MAOT / 5
    
    MAOT_var = MAOT_training_forget_0 - avg_MAOT
    MAOT_var = np.square(MAOT_var)
    MAOT_var2 = MAOT_training_forget_1 - avg_MAOT
    MAOT_var2 = np.square(MAOT_var2)
    MAOT_var3 = MAOT_training_forget_2 - avg_MAOT
    MAOT_var3 = np.square(MAOT_var3)
    MAOT_var4 = MAOT_training_forget_3 - avg_MAOT
    MAOT_var4 = np.square(MAOT_var4)
    MAOT_var5 = MAOT_training_forget_4 - avg_MAOT
    MAOT_var5 = np.square(MAOT_var5)
    
    MAOT_std_dev = MAOT_var + MAOT_var2 + MAOT_var3 + MAOT_var4 + MAOT_var5
    MAOT_std_dev = MAOT_std_dev/5
    MAOT_std_dev = np.sqrt(MAOT_std_dev)

    
    plt.plot(x,avg_MAOT_easy*100, color='purple', label='eco-system accuracy')
    plt.fill_between(x,np.clip(avg_MAOT_easy-MAOT_std_dev_easy,0,1)*100, np.clip(avg_MAOT_easy+MAOT_std_dev_easy,0,1)*100,facecolor="purple", color='purple',alpha=0.2)  
    plt.plot(x,avg_OAMT_easy*100, color='orange', label='one-agent accuracy')
    plt.fill_between(x,np.clip(avg_OAMT_easy-OAMT_std_dev_easy,0,1)*100, np.clip(avg_OAMT_easy+OAMT_std_dev_easy,0,1)*100,facecolor="orange", color='orange',alpha=0.2)  
    orange_patch = mpatches.Patch(color='orange', label='single-agent DDQN')
    purple_patch = mpatches.Patch(color='purple', label='eco-system DDQN')
    plt.plot(x,avg_MAOT*100, color='blue', label='eco-system accuracy')
    plt.fill_between(x,np.clip(avg_MAOT-MAOT_std_dev,0,1)*100, np.clip(avg_MAOT+MAOT_std_dev,0,1)*100,facecolor="blue", color='blue',alpha=0.2)  
    plt.plot(x,avg_OAMT*100, color='green', label='one-agent accuracy')
    plt.fill_between(x,np.clip(avg_OAMT-OAMT_std_dev,0,1)*100, np.clip(avg_OAMT+OAMT_std_dev,0,1)*100,facecolor="green", color='green',alpha=0.2)  
    green_patch = mpatches.Patch(color='green', label='single-agent PPO')
    blue_patch = mpatches.Patch(color='blue', label='eco-system PPO')
    plt.legend(handles=[orange_patch,purple_patch,blue_patch,green_patch])
    plt.savefig('Architecture_comparison_forget_easy.pdf')
    plt.close()


    plt.title("Generalizability index on new environments", size=16,y=1.06)
    x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    plt.xticks([1,4,8,12,16,20],[50, 200,400,600,800,1000],rotation=0,size=15)
    plt.yticks(size=15)
    plt.ylabel('% solved', size = 15)
    plt.xlabel('# of environment trained', size = 14)
    OAMT_training_general_0_easy = np.load('./Submarine-game-easy/One_agent_multi_training_general0.npy')
    MAOT_training_general_0_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_general0.npy')
    OAMT_training_general_1_easy = np.load('./Submarine-game-easy/One_agent_multi_training_general1.npy')
    MAOT_training_general_1_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_general1.npy')
    OAMT_training_general_2_easy = np.load('./Submarine-game-easy/One_agent_multi_training_general2.npy')
    MAOT_training_general_2_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_general2.npy')
    OAMT_training_general_3_easy = np.load('./Submarine-game-easy/One_agent_multi_training_general3.npy')
    MAOT_training_general_3_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_general3.npy')
    OAMT_training_general_4_easy = np.load('./Submarine-game-easy/One_agent_multi_training_general4.npy')
    MAOT_training_general_4_easy = np.load('./Submarine-game-easy/Multi_agent_one_training_general4.npy')
    OAMT_training_general_0_easy = np.delete(OAMT_training_general_0_easy,0)
    MAOT_training_general_0_easy = np.delete(MAOT_training_general_0_easy,0)
    OAMT_training_general_1_easy = np.delete(OAMT_training_general_1_easy,0)
    MAOT_training_general_1_easy = np.delete(MAOT_training_general_1_easy,0)
    OAMT_training_general_2_easy = np.delete(OAMT_training_general_2_easy,0)
    MAOT_training_general_2_easy = np.delete(MAOT_training_general_2_easy,0)
    OAMT_training_general_3_easy = np.delete(OAMT_training_general_3_easy,0)
    MAOT_training_general_3_easy = np.delete(MAOT_training_general_3_easy,0)
    OAMT_training_general_4_easy = np.delete(OAMT_training_general_4_easy,0)
    MAOT_training_general_4_easy = np.delete(MAOT_training_general_4_easy,0)
    
    min_OAMT_easy = np.minimum(OAMT_training_general_0_easy,OAMT_training_general_1_easy)
    min_OAMT_easy = np.minimum(min_OAMT_easy,OAMT_training_general_2_easy)
    min_OAMT_easy = np.minimum(min_OAMT_easy,OAMT_training_general_3_easy)
    min_OAMT_easy = np.minimum(min_OAMT_easy,OAMT_training_general_4_easy)

    max_OAMT_easy = np.maximum(OAMT_training_general_0_easy,OAMT_training_general_1_easy)
    max_OAMT_easy = np.maximum(max_OAMT_easy,OAMT_training_general_2_easy)
    max_OAMT_easy = np.maximum(max_OAMT_easy,OAMT_training_general_3_easy)
    max_OAMT_easy = np.maximum(max_OAMT_easy,OAMT_training_general_4_easy) 
    
    avg_OAMT_easy = OAMT_training_general_0_easy + OAMT_training_general_1_easy + OAMT_training_general_2_easy + OAMT_training_general_3_easy + OAMT_training_general_4_easy
    avg_OAMT_easy = avg_OAMT_easy / 5 
        
    OAMT_var_easy = OAMT_training_general_0_easy - avg_OAMT_easy
    OAMT_var_easy = np.square(OAMT_var_easy)
    OAMT_var2_easy = OAMT_training_general_1_easy - avg_OAMT_easy
    OAMT_var2_easy = np.square(OAMT_var2_easy)
    OAMT_var3_easy = OAMT_training_general_2_easy - avg_OAMT_easy
    OAMT_var3_easy = np.square(OAMT_var3_easy)
    OAMT_var4_easy = OAMT_training_general_3_easy - avg_OAMT_easy
    OAMT_var4_easy = np.square(OAMT_var4_easy)
    OAMT_var5_easy = OAMT_training_general_4_easy - avg_OAMT_easy
    OAMT_var5_easy = np.square(OAMT_var5_easy)
    
    OAMT_std_dev_easy = OAMT_var_easy + OAMT_var2_easy + OAMT_var3_easy + OAMT_var4_easy + OAMT_var5_easy
    OAMT_std_dev_easy = OAMT_std_dev_easy/5
    OAMT_std_dev_easy = np.sqrt(OAMT_std_dev_easy)

    min_MAOT_easy = np.minimum(MAOT_training_general_0_easy,MAOT_training_general_1_easy)
    min_MAOT_easy = np.minimum(min_MAOT_easy,MAOT_training_general_2_easy)
    min_MAOT_easy = np.minimum(min_MAOT_easy,MAOT_training_general_3_easy)
    min_MAOT_easy = np.minimum(min_MAOT_easy,MAOT_training_general_4_easy)

    max_MAOT_easy = np.maximum(MAOT_training_general_0_easy,MAOT_training_general_1_easy)
    max_MAOT_easy = np.maximum(max_MAOT_easy,MAOT_training_general_2_easy)
    max_MAOT_easy = np.maximum(max_MAOT_easy,MAOT_training_general_3_easy)
    max_MAOT_easy = np.maximum(max_MAOT_easy,MAOT_training_general_4_easy)   
    
    avg_MAOT_easy = MAOT_training_general_0_easy + MAOT_training_general_1_easy + MAOT_training_general_2_easy + MAOT_training_general_3_easy + MAOT_training_general_4_easy
    avg_MAOT_easy = avg_MAOT_easy / 5
    
    MAOT_var_easy = MAOT_training_general_0_easy - avg_MAOT_easy
    MAOT_var_easy = np.square(MAOT_var_easy)
    MAOT_var2_easy = MAOT_training_general_1_easy - avg_MAOT_easy
    MAOT_var2_easy = np.square(MAOT_var2_easy)
    MAOT_var3_easy = MAOT_training_general_2_easy - avg_MAOT_easy
    MAOT_var3_easy = np.square(MAOT_var3_easy)
    MAOT_var4_easy = MAOT_training_general_3_easy - avg_MAOT_easy
    MAOT_var4_easy = np.square(MAOT_var4_easy)
    MAOT_var5_easy = MAOT_training_general_4_easy - avg_MAOT_easy
    MAOT_var5_easy = np.square(MAOT_var5_easy)
    
    MAOT_std_dev_easy = MAOT_var_easy + MAOT_var2_easy + MAOT_var3_easy + MAOT_var4_easy + MAOT_var5_easy
    MAOT_std_dev_easy = MAOT_std_dev_easy/5
    MAOT_std_dev_easy = np.sqrt(MAOT_std_dev_easy)
    
    OAMT_training_general_0 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_general0.npy')
    MAOT_training_general_0 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_general0.npy')
    OAMT_training_general_1 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_general1.npy')
    MAOT_training_general_1 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_general1.npy')
    OAMT_training_general_2 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_general2.npy')
    MAOT_training_general_2 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_general2.npy')
    OAMT_training_general_3 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_general3.npy')
    MAOT_training_general_3 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_general3.npy')
    OAMT_training_general_4 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_general4.npy')
    MAOT_training_general_4 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_general4.npy')
    OAMT_training_general_0 = np.delete(OAMT_training_general_0,0)
    MAOT_training_general_0 = np.delete(MAOT_training_general_0,0)
    OAMT_training_general_1 = np.delete(OAMT_training_general_1,0)
    MAOT_training_general_1 = np.delete(MAOT_training_general_1,0)
    OAMT_training_general_2 = np.delete(OAMT_training_general_2,0)
    MAOT_training_general_2 = np.delete(MAOT_training_general_2,0)
    OAMT_training_general_3 = np.delete(OAMT_training_general_3,0)
    MAOT_training_general_3 = np.delete(MAOT_training_general_3,0)
    OAMT_training_general_4 = np.delete(OAMT_training_general_4,0)
    MAOT_training_general_4 = np.delete(MAOT_training_general_4,0)
    
    min_OAMT = np.minimum(OAMT_training_general_0,OAMT_training_general_1)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_general_2)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_general_3)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_general_4)
    
    max_OAMT = np.maximum(OAMT_training_general_0,OAMT_training_general_1)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_general_2)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_general_3)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_general_4) 
    
    avg_OAMT = OAMT_training_general_0 + OAMT_training_general_1+OAMT_training_general_2+OAMT_training_general_3+OAMT_training_general_4
    avg_OAMT = avg_OAMT / 5 
        
    OAMT_var = OAMT_training_general_0 - avg_OAMT
    OAMT_var = np.square(OAMT_var)
    OAMT_var2 = OAMT_training_general_1 - avg_OAMT
    OAMT_var2 = np.square(OAMT_var2)
    OAMT_var3 = OAMT_training_general_2 - avg_OAMT
    OAMT_var3 = np.square(OAMT_var3)
    OAMT_var4 = OAMT_training_general_3 - avg_OAMT
    OAMT_var4 = np.square(OAMT_var4)
    OAMT_var5 = OAMT_training_general_4 - avg_OAMT
    OAMT_var5 = np.square(OAMT_var5)
    
    OAMT_std_dev = OAMT_var + OAMT_var2 + OAMT_var3 + OAMT_var4 + OAMT_var5
    OAMT_std_dev = OAMT_std_dev/5
    OAMT_std_dev = np.sqrt(OAMT_std_dev)
    
    min_MAOT = np.minimum(MAOT_training_general_0,MAOT_training_general_1)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_general_2)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_general_3)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_general_4)
    
    max_MAOT = np.maximum(MAOT_training_general_0,MAOT_training_general_1)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_general_2)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_general_3)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_general_4)   
    
    avg_MAOT = MAOT_training_general_0 + MAOT_training_general_1+MAOT_training_general_2+MAOT_training_general_3+MAOT_training_general_4
    avg_MAOT = avg_MAOT / 5
    
    MAOT_var = MAOT_training_general_0 - avg_MAOT
    MAOT_var = np.square(MAOT_var)
    MAOT_var2 = MAOT_training_general_1 - avg_MAOT
    MAOT_var2 = np.square(MAOT_var2)
    MAOT_var3 = MAOT_training_general_2 - avg_MAOT
    MAOT_var3 = np.square(MAOT_var3)
    MAOT_var4 = MAOT_training_general_3 - avg_MAOT
    MAOT_var4 = np.square(MAOT_var4)
    MAOT_var5 = MAOT_training_general_4 - avg_MAOT
    MAOT_var5 = np.square(MAOT_var5)
    
    MAOT_std_dev = MAOT_var + MAOT_var2 + MAOT_var3 + MAOT_var4 + MAOT_var5
    MAOT_std_dev = MAOT_std_dev/5
    MAOT_std_dev = np.sqrt(MAOT_std_dev)

         
    plt.plot(x,avg_MAOT_easy, color='purple', label='eco-system accuracy')
    plt.fill_between(x,np.clip(avg_MAOT_easy-MAOT_std_dev_easy,0,100), np.clip(avg_MAOT_easy+MAOT_std_dev_easy,0,100),facecolor="purple", color='purple',alpha=0.2)  
    plt.plot(x,avg_OAMT_easy, color='orange', label='one-agent accuracy')
    plt.fill_between(x,np.clip(avg_OAMT_easy-OAMT_std_dev_easy,0,100), np.clip(avg_OAMT_easy+OAMT_std_dev_easy,0,100),facecolor="orange", color='orange',alpha=0.2)  
    orange_patch = mpatches.Patch(color='orange', label='single-agent DDQN')
    purple_patch = mpatches.Patch(color='purple', label='eco-system DDQN')
    plt.plot(x,avg_MAOT, color='blue', label='eco-system accuracy')
    plt.fill_between(x,np.clip(avg_MAOT-MAOT_std_dev,0,100), np.clip(avg_MAOT+MAOT_std_dev,0,100),facecolor="blue", color='blue',alpha=0.2)  
    plt.plot(x,avg_OAMT, color='green', label='one-agent accuracy')
    plt.fill_between(x,np.clip(avg_OAMT-OAMT_std_dev,0,100), np.clip(avg_OAMT+OAMT_std_dev,0,100),facecolor="green", color='green',alpha=0.2)  
    green_patch = mpatches.Patch(color='green', label='single-agent PPO')
    blue_patch = mpatches.Patch(color='blue', label='eco-system PPO')
    plt.legend(handles=[orange_patch,purple_patch,blue_patch,green_patch])
    plt.savefig('Architecture_comparison_general_easy.pdf')
    plt.close()
    
    plt.title("Number of accesses to environments", size=16,y=1.06)
    x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    plt.xticks([1,4,8,12,16,20],[50, 200,400,600,800,1000],rotation=0,size=15)
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
    
    OAMT_training_access_0 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_access0.npy')
    MAOT_training_access_0 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_access0.npy')
    OAMT_training_access_1 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_access1.npy')
    MAOT_training_access_1 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_access1.npy')
    OAMT_training_access_2 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_access2.npy')
    MAOT_training_access_2 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_access2.npy')
    OAMT_training_access_3 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_access3.npy')
    MAOT_training_access_3 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_access3.npy')
    OAMT_training_access_4 = np.load('./Submarine-game-easy-PPO/One_agent_multi_training_access4.npy')
    MAOT_training_access_4 = np.load('./Submarine-game-easy-PPO/Multi_agent_one_training_access4.npy')
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
    
     
    plt.plot(x,avg_MAOT_easy, color='purple', label='eco-system access')
    plt.fill_between(x,avg_MAOT_easy-MAOT_std_dev_easy, avg_MAOT_easy+MAOT_std_dev_easy,facecolor="purple", color='purple',alpha=0.2)  
    plt.plot(x,avg_OAMT_easy, color='orange', label='one-agent access')
    plt.fill_between(x,avg_OAMT_easy-OAMT_std_dev_easy, avg_OAMT_easy+OAMT_std_dev_easy,facecolor="orange", color='orange',alpha=0.2)  
    orange_patch = mpatches.Patch(color='orange', label='single-agent DDQN')
    purple_patch = mpatches.Patch(color='purple', label='eco-system DDQN')
    plt.legend(handles=[orange_patch,purple_patch])
    plt.savefig('Architecture_comparison_access_easy.pdf')
    plt.close()
    

def main():
    generate_results_comparison()

if __name__ == "__main__":
    main()
