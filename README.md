# BNPO : Bellman with Natural and Policy Optimization
* Related blogpost : [B(N)PO : Combined Off and On Policy Algorithm](https://rezer0dai.github.io/bnpo/)
  - introducing new RL algorithm (BNPO)
  - Combine On-policy and Off-policy
  - Combines PPO and HER
  - briefly experiment with Natural Gradients for D(D)PG

More Information
===
check also related [research notes](https://rezer0dai.github.io/rl-notes/) and older [blog](https://rezer0dai.github.io/rewheeler/)

### Environment setup from [older project](https://github.com/rezer0dai/ReacherBenchmarkDDPG) same [environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md)
  - state_size=33, action_size=4 as default UnityML - Reacher environment provides
  - 20 arms environment, used shared actor + critic for controlling all arms
  - How to install :
  - environment itself : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
    - unpack package to ./reach/ folder inside this project

  - replicate results by running [notebooks](https://rezer0dai.github.io/bnpo/#jupyter-notebooks)

### necessary pip packages : 
  - pytorch
  - unityagents
  - timebudget
  - mathplotlib
