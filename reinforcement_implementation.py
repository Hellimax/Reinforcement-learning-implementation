import numpy as np 

R = np.matrix([[-1,-1,-1,-1,0,-1], #The reward matrix
                [-1,-1,-1,0,-1,100],
                [-1,-1,-1,0,-1,-1],
                [-1,0,0,-1,0,-1],
                [0,-1,-1,0,-1,100],
                [-1,0,-1,-1,0,100]])

Q = np.zeros((6,6)) #the Q-matrix
gamma = 0.8 #setting the gamma value

def action_to_perform(state): #this is to find what steps can be taken by our agent to go to nect state  
    current_state_row = R[state,]#for eg if the agent is at state 1 it can only go to state 3 and 5
    av_ac = np.where(current_state_row>=0)[1]
    action = np.random.choice(list(av_ac))
    return(action,av_ac)

def update_q(current_state,action): #this is to update Q-matrix
    _,action_list = action_to_perform(action)
    max_value = max(Q[action,action_list])
    Q[current_state,action] = R[current_state,action]+gamma*max_value

def check_model(): #this is to check our model after training
    current_state = int(input("Enter the current state "))
    path = [current_state]
    while current_state!=5:
        next_step = np.where(Q[current_state,]==np.max(Q[current_state,]))[0]
        if next_step.shape[0]>1:
            next_step = int(np.random.choice(next_step))
        else:
            next_step = int(next_step)
        path.append(next_step)
        current_state = next_step
    return path
    

if __name__ == "__main__":
    for i in range(1000): #here we are training our model by 1000 iterations
        current_state = np.random.randint(0,5)
        action,_=action_to_perform(current_state)
        update_q(current_state,action)
        print("Iteration "+str(i)+" performed")
    final_q = Q/Q.max()*100
    print(final_q)
    path = check_model()
    print("the selected path is ",path)

