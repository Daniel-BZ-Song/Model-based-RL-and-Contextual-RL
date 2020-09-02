# Model-based Reinforcement Learning

Here we summarized the code and the report for Summer reseach project.

## Run the test

Run Case_multiposs.py to run the simulation via multiprocessor, current version take 32 cores and run all 32 paths in one run.
> python Case_multiposs.py
The parameter can be passed through *para_ls* variable, which can hold multiple tuples of parameters. In each tuple the the parameters contain:
* **ENV_TYPE**: Type of MDP structure. Currently support "River" (River Swim MDP) or "Tree" (Tree MDP)
* **ACTION**: Cardinality of action space. If ENV_TYPE="River", then default 2 actions; If ENV_TYPE="Tree MDP", then number of action should be 2n+1 for n >1.
* **STATE**: Cardinality of state space. Should be larger than 3.
* **TOTAL_TIME**: Total iteration time T for algorithm.
* **BAYESIAN**: Bayesian Setting for the environment. If "True", every new environment intance's transition probability will be sampled from a prior distribution. If "False", the transition probability fixed for all instances. 

## Empirical Result
* The Thompson Sampling method PSRL achieves a significantly lower total regret and faster converge speed compared with UCRL2 algorithm in both small and large states space. 
* When the state space is small, PSRL with only single sampling achieves a slight lower total regret compared with PSRL with multi-sampling. However, when the state space becomes large, single sampling algorithm may not converge. 

