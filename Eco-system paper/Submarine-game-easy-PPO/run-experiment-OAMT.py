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
            
            
class OneAgentMultipleTrainings():
    def __init__ (self):
        self.single_agent = SubMarineAgent(0)
        self.nb_access = 0 
        
    def get_access(self):
        return self.nb_access
    
    def reset_access(self):
        self.nb_access=0

    def train(self,env_seed):
        print("Training on environment ",env_seed)
        start_time = time.time()
        self.single_agent.change_environment(env_seed)
        self.single_agent.reset_access()
        if self.single_agent.check()<100:
            self.single_agent.solve()
        self.nb_access+=self.single_agent.get_access()
        elapsed_time = time.time() - start_time
        print("Time elapsed: ",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        sys.stdout.flush()
        return elapsed_time

    def checkpoint(self,it):
        self.single_agent.save(0,it)

    def restore(self,it):
        self.single_agent = Agent(0)
        self.single_agent.load(0,it)		

    def test(self,nb_env):
        self.single_agent.reset_access()
        self.single_agent.change_environment(nb_env)
        res = self.single_agent.check()
        self.nb_access+=self.single_agent.get_access()
        return res

def generate_results_OAMT():
    for cpt in range(5):
        rnd = np.random.randint(650000)
        np.random.seed(rnd)
        random.seed(rnd) 
        OAMT=OneAgentMultipleTrainings()
        training_time=[]
        forget = []
        general=[]
        access = []
        t_time=0
        for i in range(0,1001):
            t_time+=OAMT.train(i)
            if i%50==0:
                training_time.append(t_time)
                access.append(OAMT.get_access())
                nb_ok=0
                for j in range(0,i):
                    if OAMT.test(j)>=100:
                        nb_ok+=1
                if i>0:
                    forget.append(nb_ok/i)
                else:
                    forget.append(nb_ok)
                nb_ok=0
                for j in range(1000,2000):
                    if OAMT.test(j)>=100:
                        nb_ok+=1
                general.append(nb_ok/10)
        tt_npy = np.array(training_time)
        np.save('One_agent_multi_training_time'+str(cpt)+'.npy',tt_npy)
        f_npy=np.array(forget)
        np.save('One_agent_multi_training_forget'+str(cpt)+'.npy',f_npy)
        g_npy=np.array(general)
        np.save('One_agent_multi_training_general'+str(cpt)+'.npy',g_npy)
        a_npy=np.array(access)
        np.save('One_agent_multi_training_access'+str(cpt)+'.npy',a_npy)
    
def main():
    generate_results_OAMT()

if __name__ == "__main__":
    main()
