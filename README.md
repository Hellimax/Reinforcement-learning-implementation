# Reinforcement-learning-implementation
this is an example to implement Q-learning, a reinforcement learning algorithm, in python using only numpy

modules needed == Numpy

SO this is just an example on how you can implement Q-learning in python using numpy only.
We are assuming we already have a reward matrix R of 6X6 and a Q matrix of 6X6, at the begning the Q-matrix is 0 

Let us assume we have 6 states 0,1,2,3,4,5 and 5 is our final state, if a state has a direct connection to 5 then it has a reward point of 100
, if a state has a direct connection to any other state other then 5 then it has a reward point of 0, otherwise the reward point is -1

so our main goal here to find a path from any state to state 5 using reinforcemnent learning.
