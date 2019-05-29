#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import random
import queue

#0-Rock;1-Scissor;2-Paper
def strategy(state):
    AI_list= []
    in_file = 'RESULT.csv'
    full_data = pd.read_csv(in_file)
    AI_list = full_data['AI']
    
    count_0 = count_1 = count_2 = 0 #times of 0,1,2 appears respectively 
    count_00 = count_01 = count_02 = 0 # transfer times starting with 0
    count_10 = count_11 = count_12 = 0 # transfer times starting with 1
    count_20 = count_21 = count_22 = 0 # transfer times starting with 2
    p_00 = p_01 = p_02 = 0 # transfer probability starting with 0
    p_10 = p_11 = p_12 = 0 # transfer probability starting with 1
    p_20 = p_21 = p_22 = 0 # transfer probability starting with 2
    
    for i in range(len(AI_list)):
        if(i+1<100):
            if (AI_list[i] == 0):
                count_0 += 1
                if (AI_list[i+1] == 0): count_00 += 1
                if (AI_list[i+1] == 1): count_01 += 1
                if (AI_list[i+1] == 2): count_02 += 1
            elif (AI_list[i] == 1):
                count_1 += 1
                if (AI_list[i+1] == 0): count_10 += 1
                if (AI_list[i+1] == 1): count_11 += 1
                if (AI_list[i+1] == 2): count_12 += 1
            elif (AI_list[i] == 2):
                count_2 += 1
                if (AI_list[i+1] == 0): count_20 += 1
                if (AI_list[i+1] == 1): count_21 += 1
                if (AI_list[i+1] == 2): count_22 += 1
    
    p_00 = count_00 / count_0 #transfer probability
    p_01 = count_01 / count_0
    p_02 = count_02 / count_0
    p_10 = count_10 / count_1
    p_11 = count_11 / count_1
    p_12 = count_12 / count_1
    p_20 = count_20 / count_2
    p_21 = count_21 / count_2
    p_22 = count_22 / count_2
    
    random1 = random.randint(1,100)
    if (state == 1):
        if (random1 < (p_10*100)):
            state = 0
            nextmove = 2
        if ((p_10*100) < random1 < (p_10*100 + p_11*100)): 
            state = 1
            nextmove = 0
        if ((p_10*100 + p_11*100) < random1 <= 100): 
            state = 2
            nextmove = 1
    
    elif (state == 0):
        if (random1 < (p_00*100)): 
            state = 0
            nextmove = 2
        if ((p_00*100) < random1 < (p_00*100 + p_01*100)): 
            state = 1
            nextmove = 0
        if ((p_00*100 + p_01*100) < random1 <= 100): 
            state = 2
            nextmove = 1
    
    elif (state == 2):
        if (random1 < (p_20*100)): 
            state = 0
            nextmove = 2
        if ((p_20*100) < random1 < (p_20*100 + p_21*100)): 
            state = 1
            nextmove = 0
        if ((p_20*100 + p_21*100) < random1 <= 100): 
            state = 2
            nextmove = 1
            
    else: nextmove = random.randint(0,2)
    
    return nextmove

def advanced(state, AI_move):
    if ((state == AI_move.get()) and (state == AI_move.get())):
        if(state == 0): res = random.randint(1,2)
        if(state == 1): res = random.randrange(0,2)
        if(state == 2): res = random.randint(0,1)
        return res
    else: strategy(state)
 
if __name__ == "__main__":
    AI_move = queue.Queue(maxsize=2) #copy of AI_move to trigger def advanced()
    AI_move2 = queue.Queue(maxsize=2)
    while(True):
        state = int(input("Current state of AI? 0-Rock,1-Scissor,2-Paper,Else-Random:"))
        if (AI_move.full() == True):
            AI_move.get()
            AI_move.put(state)
        else: AI_move.put(state)
            
        if (AI_move2.full() == True):
            AI_move2.get()
            AI_move2.put(state)
        else: AI_move2.put(state)
        
        if (AI_move.full() == True):
            res = advanced(state, AI_move)
        else: res = strategy(state)
        if (res == 0): print("Your best move is: Rock")
        if (res == 1): print("Your best move is: Scissor")
        if (res == 2): print("Your best move is: Paper")