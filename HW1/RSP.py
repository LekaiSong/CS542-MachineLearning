#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import random
#from transitions import Machine

#0-Rock;1-Scissor;2-Paper
def analyze():
    AI_list= []
    in_file = 'RESULT.csv'
    full_data = pd.read_csv(in_file)
    AI_list = full_data['AI']
    #data = full_data.drop()
    
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
    
    p_00 = count_00 / count_0
    p_01 = count_01 / count_0
    p_02 = count_02 / count_0
    p_10 = count_10 / count_1
    p_11 = count_11 / count_1
    p_12 = count_12 / count_1
    p_20 = count_20 / count_2
    p_21 = count_21 / count_2
    p_22 = count_22 / count_2

    global nextmove #initial
    #while(True):
    state = input("Current state of AI? 0-Rock,1-Scissor,2-Paper:")
    state = int(state)
    #print(type(state))
    x = 1
    L = []
    sum = 0
    while x < 4:
        L.append(random.randint(1,100))
        x += 1
    for n in L:
        sum += n
    random1 = sum/3
    print(random1)
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
    

if __name__ == "__main__":
    print("The best move is: " + str(analyze()))
    
#def statemachine():
#    class Matter(object):
#        pass
#    model = Matter()
#    
#    states=['0', '1', '2']
#    transitions = [
#            {'trigger': 'p_00', 'source': '0', 'dest': '0'},
#            {'trigger': 'p_01', 'source': '0', 'dest': '1'},
#            {'trigger': 'p_02', 'source': '0', 'dest': '2'},
#            {'trigger': 'p_10', 'source': '1', 'dest': '0'},
#            {'trigger': 'p_11', 'source': '1', 'dest': '1'},
#            {'trigger': 'p_12', 'source': '1', 'dest': '2'},
#            {'trigger': 'p_20', 'source': '2', 'dest': '0'},
#            {'trigger': 'p_21', 'source': '2', 'dest': '1'},
#            {'trigger': 'p_22', 'source': '2', 'dest': '2'}]
#    
#    machine = Machine(model=model, states=states, transitions=transitions, initial='0')
#    return model.state
