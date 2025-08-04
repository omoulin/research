# Eco-system paper - (c) 2021 Olivier Moulin, Amsterdam Vrije Universiteit 

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/.

import numpy as np
from datetime import datetime
from scipy.special import softmax
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
import gym
from gym import spaces
from PIL import Image
from stable_baselines3 import PPO

import time


length = 20

class MarineEnv(gym.Env):
    def __init__(self):
        self._last_ponctual_observation = [0, 0, 0]
        self.seed = 1
        self.Marine=np.load('level/level-'+str(self.seed)+'.npy')
        self.X=0
        self.Y=5
        self.length=length
        self.Y_history = []
        self.Y_history.append(5)
        self.total_step=0
        self.display_submarine = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                             [1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                             [1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                            ]
        self.display_rock = []
        for i in range(0,20):
            line = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            if i<5:
                for j in range(0,20):
                    line[j]=np.random.randint(0,2)
            else:
                for j in range(0,5):
                    line[j]=np.random.randint(0,2)
                for j in range(15,20):
                    line[j]=np.random.randint(0,2)
            self.display_rock.append(line)
        self.environment_access = 0
        self.action_space = spaces.Discrete(3,)
        self.observation_space = spaces.Box(low=0, high=255,shape=(11*5+1,), dtype=np.float32)

    def step(self, action):
        reward = 1
        done = False
        self.total_step = self.total_step+1
        self.X=self.X+1
        if action==0:
            self.Y=self.Y-1
        if action==1:
            self.Y=self.Y+1
        if action==2:
            self.Y=self.Y
        if self.Marine[self.Y,self.X]==1:
            reward=-100
            done = True
        else:
            if self.X>=self.length-1:
                reward=100
                done= True
        self.Y_history.append(self.Y)
        res=np.zeros((11*5+1,))
        res[0:11*5]=self.Marine[0:11,self.X:self.X+5].flatten()
        res[11*5]=self.Y
        state=res
        return state,reward,done,{}

    def reset(self):
        self.X=0
        self.Y=5
        res=np.zeros((11*5+1,))
        res[0:11*5]=self.Marine[0:11,self.X:self.X+5].flatten()
        res[11*5]=self.Y
        state=res
        self.total_step=0
        self.Y_history=[]
        self.Y_history.append(5)
        return state

    def reset_access(self):
        self.environment_access=0

    def get_access(self):
        return self.environment_access

    def update_seed (self,seed):
        self.seed = seed
        self.Marine=np.load('level/level-'+str(seed)+'.npy')
        self.reset()



class SubMarineAgent():            
        
        def __init__ (self,env_nb):
            self.env = MarineEnv()
            self.env.update_seed(env_nb)	
            self.env.reset()	
            #self.print_environment()
            self.model = PPO("MlpPolicy",self.env,verbose=0)
            self.nb_access = 0
            
        def change_environment(self,nb_env):
            self.env.update_seed(nb_env)
        
        def check(self):
            state = self.env.reset()
            total_rew =0
            done = False
            self.env.reset_access()
            while done == False:
                action, _states = self.model.predict(state,deterministic=True)
                state, reward, done, _ = self.env.step(action)
                total_rew+=reward
            self.nb_access+=self.env.get_access()
            return total_rew
        
        def solve(self):
            training_frame = 0
            solved = False
            while solved==False:
                self.env.reset_access()
                self.model.learn(total_timesteps=10000)
                training_frame+=10000
                self.nb_access+=self.env.get_access()
                self.env.reset_access()
                chk = self.check()
                self.nb_access+=self.env.get_access()
                print("Check for stopping training : ",chk)
                if chk>=0.80:
                    solved = True
                sys.stdout.flush()
            return training_frame
                
        def save(self,i,it):
            self.model.save('Agent_'+str(i)+'_'+str(it)+'.mdl')
        
        def load(self,i,it):
            self.model = PPO.load('Agent_'+str(i)+'_'+str(it)+'.mdl')
                    
        def reset_access(self):
            self.nb_access=0
            
        def get_access(self):
            return self.nb_access


class MultipleAgentsOneTraining():
    def __init__ (self):
        self.Agent_grid=[]
        self.nb_access=0
        i=0

    def get_access(self):
        return self.nb_access

    def reset_access(self):
        self.nb_access=0

    def sort_list(self,e):
        return(e[1].size)

    def train(self,env_seed):
        training_frame = 0
        print("Training on environment ",env_seed)
        start_time = time.time()
        solved= False
        i = 0
        while solved==False and i<len(self.Agent_grid):
            ag_test = self.Agent_grid[i][0]
            ag_test.change_environment(env_seed)
            ag_test.reset_access()
            if ag_test.check()>=100:
                solved=True
            else:
                i+=1
            self.nb_access+=ag_test.get_access()
        if solved==False:
            ag=SubMarineAgent(env_seed)
            ag.reset_access()
            ag.solve()
            self.nb_access+=ag.get_access()
            grid_ag = (ag,np.array([env_seed]))
            self.Agent_grid.append(grid_ag)
            i = len(self.Agent_grid)-1
            k = 0
            while k<len(self.Agent_grid):
                list_env = self.Agent_grid[k][1].tolist()
                can_solve_all=True
                for env_test in list_env:
                    self.Agent_grid[i][0].change_environment(env_test)
                    self.Agent_grid[i][0].reset_access()
                    if self.Agent_grid[i][0].check()>=100:
                        if env_test not in self.Agent_grid[i][1]:
                            self.Agent_grid[i]=(self.Agent_grid[i][0],np.append(self.Agent_grid[i][1],env_test))
                    else:
                        can_solve_all=False
                    self.nb_access+=self.Agent_grid[i][0].get_access()
                if can_solve_all==True and k!=i:
                    self.Agent_grid.pop(k)
                    if k<i:
                        i-=1
                else:
                    k+=1
        else:
            if env_seed not in self.Agent_grid[i][1]:
                self.Agent_grid[i]=(self.Agent_grid[i][0],np.append(self.Agent_grid[i][1],env_seed))
        self.Agent_grid.sort(key=self.sort_list,reverse=True)
        elapsed_time = time.time() - start_time
        print("Time elapsed: ",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        sys.stdout.flush()
        return elapsed_time

    def print_architecture(self):
        print("Agent eco-system architecture")
        print("Number of active agents : ",len(self.Agent_grid))
        total_env = 0
        for i in range(0,len(self.Agent_grid)):
            total_env += self.Agent_grid[i][1].size
            print("Number of environments covered : ",total_env)
        print("Architecture")
        for i in range(0,len(self.Agent_grid)):
            print("Agent : ",i)
            print("Environments : ",self.Agent_grid[i][1])

    def checkpoint(self,it):
        for i in range(0,len(self.Agent_grid)):
            self.Agent_grid[i][0].save(i,it)
            np.save('Agent_covered_'+str(i)+'_'+str(it)+'.npy',self.Agent_grid[i][1])

    def restore(self,it):
        self.Agent_grid=[]
        for i in range(0,len(self.Agent_grid)):
            ag = SubMarineAgent(0)
            ag.load(i,it)
            env_covered = np.load('Agent_covered_'+str(i)+'_'+str(it)+'.npy')
            self.Agent_grid.append((ag,env_covered))
            
    def test(self,nb_env):
        max_res=0
        found = False
        j=0
        while found == False and j<len(self.Agent_grid):
            if nb_env in self.Agent_grid[j][1]:
                found = True
                self.Agent_grid[j][0].change_environment(nb_env)
                self.Agent_grid[j][0].reset_access()
                max_res = self.Agent_grid[j][0].check()
                self.nb_access+=self.Agent_grid[j][0].get_access()
            j+=1
        j=0
        while found == False and j<len(self.Agent_grid):
            self.Agent_grid[j][0].change_environment(nb_env)
            self.Agent_grid[j][0].reset_access()
            res=self.Agent_grid[j][0].check()
            self.nb_access+=self.Agent_grid[j][0].get_access()
            if res>max_res:
                max_res=res
            if res>=100:
                found = True
            j+=1
        return max_res
    
def generate_results_MAOT():
    for cpt in range(5):
        rnd = np.random.randint(650000)
        np.random.seed(rnd)
        random.seed(rnd) 
        MAOT=MultipleAgentsOneTraining()
        training_time=[]
        forget = []
        general=[]
        access=[]
        t_time=0
        for i in range(0,1001):
            t_time+=MAOT.train(i)
            if i%50==0:
                training_time.append(t_time)
                access.append(MAOT.get_access())
                nb_ok=0
                for j in range(0,i):
                    if MAOT.test(j)>=100:
                        nb_ok+=1
                if i>0:
                    forget.append(nb_ok/i)
                else:
                    forget.append(nb_ok)
                nb_ok=0
                for j in range(1000,2000):
                    if MAOT.test(j)>=100:
                        nb_ok+=1
                general.append(nb_ok/10)
        tt_npy = np.array(training_time)
        np.save('Multi_agent_one_training_time'+str(cpt)+'.npy',tt_npy)
        f_npy=np.array(forget)
        np.save('Multi_agent_one_training_forget'+str(cpt)+'.npy',f_npy)
        g_npy=np.array(general)
        np.save('Multi_agent_one_training_general'+str(cpt)+'.npy',g_npy)
        a_npy=np.array(access)
        np.save('Multi_agent_one_training_access'+str(cpt)+'.npy',a_npy)

def main():
    generate_results_MAOT()

if __name__ == "__main__":
    main()
