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
np.random.seed(1231231)

import tensorflow as tf
tf.random.set_seed(1231231)

from datetime import datetime
from scipy.special import softmax
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
import gym
from PIL import Image

import time

def main():
    for seed in range(0,10000):
        np.random.seed(seed)
        Marine=np.zeros((11,25))
        Marine[0,0:20]= 1
        Marine[10,0:20]=1
        for i in range(0,25):
            if i%2==0 and i>0:
                for j in range(0,3):
                    rnd = np.random.randint(0,11)
                    Marine[rnd,i]=1
            if i>=20:
                Marine[0:11,i]=1
        np.save('level/level-'+str(seed)+'.npy',Marine)



if __name__ == "__main__":
    main()
