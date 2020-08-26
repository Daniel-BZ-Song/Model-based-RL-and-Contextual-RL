import gym
from gym import spaces
import numpy as np
import math


class RiverSwim(gym.Env):
    def __init__(self, state, sample=False):
        self._A = 2
        self._S = state

        self.state = None
        self.initial_state = 0

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(state)

        self.prob_vec_1 = np.array([0.05, 0.6, 0.35])
        self.prob_vec_2 = np.array([0, 0.4, 0.6])
        if sample:
            i = 0
            while True:
                self.prob_vec_1 = np.random.dirichlet(np.array([0.05, 0.6, 0.35]))
                i+=1
                if self.prob_vec_1[2] > self.prob_vec_1[0]*2:
                    break
                if i >20:
                    print("Sample to much time")
                    break
            i = 0
            while True:
                temp= np.random.dirichlet(np.array([0.4, 0.6]))
                i+=1
                if temp[1] > temp[0]:
                    self.prob_vec_2 = np.array([0, temp[0], temp[1]])
                    break
                if i > 20:
                    print("Sample to much time")
                    break
        self.P_s3 = np.repeat(np.array([0.05, 0.6, 0.35])[np.newaxis,:], self._S, axis=0)
        self.P_s3[0, :] = self.prob_vec_2
        self.P_s3[-1, :] = self.prob_vec_2

        self.P_mean = self.P_s3.copy()

        self.P_mean = np.repeat(self.prob_vec_1[np.newaxis, :], self._S, axis=0)
        self.P_mean[0, :] = np.array([0, 0.4, 0.6])
        self.P_mean[-1, :] = np.array([0, 0.4, 0.6])
        # if sample:
        #     self.P_s3 = self.prob_sample(self.P_s3)

    def step(self, action):
        random_unif = np.random.uniform()
        reward = 0
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            if self.state == 0 or self.state == self._S - 1:
                if random_unif <= self.P_s3[self.state, 1]:
                    self.state = max(0, self.state - 1)
                else:
                    self.state = min(self._S - 1, self.state + 1)
            else:
                if random_unif <= self.P_s3[self.state, 0]:  #0.05:
                    self.state -= 1
                elif random_unif >= 1 - self.P_s3[self.state, 1]:  #0.65:
                    self.state += 1

        if self.state == 0:
            reward = 5 / 1000
        elif self.state == self._S - 1:
            reward = 1

        return self.state, reward, False, {}

    def reset(self):
        self.state = self.initial_state
        return self.state

    def render(self, mode='human'):
        return None

    def close(self):
        return None

    def A(self):
        # Get the action space size
        return self._A

    def S(self):
        # Get the state space size
        return self._S

    def bestR(self):
        # Get the best average rewards
        return 0.45/0.8

    def Dirichlet(self, M_vec):
        # Sample from dirichlet distribution
        # If there is zero exists, skip that zero
        assert np.all(M_vec >= 0), "There is negative para in Dirichlet Distribution"
        result = np.zeros(M_vec.shape)
        idx = M_vec > 0
        result[idx] = np.random.dirichlet(M_vec[idx])
        return result

    def prob_sample(self, prob):
        # sample a transition from the deterministic probability
        sample_prob = np.apply_along_axis(self.Dirichlet, 1, prob)
        assert np.all(np.round(sample_prob.sum(axis=1), 3) == 1), "Check sampled probability"
        return sample_prob

    def get_transition(self):
        transition = np.zeros((self._S, self._A, self._S))

        for i in range(self._S):
            transition[i, 1, max(0, i-1)] = 1
            transition[i, 1, i] = self.P_mean[i, 1]
            if i == 0:
                transition[i, 1,  i+1] = self.P_mean[i, 2]
            elif i == self._S - 1:
                transition[i, 1, i-1] = self.P_mean[i, 1]
                transition[i, 1, i] = self.P_mean[i, 2]
            else:
                transition[i, 1, i-1] = self.P_mean[i, 0]
                transition[i, 1, i+1] = self.P_mean[i, 2]
        return transition




class TreeMDP(gym.Env):
    """
    Tree MDP env object

    Described in Jaksch et al. [2010], Part-6 The Lower Bound

    The initial state is the first state at the bottom level
    The good state is the bottom of the last branch
    """
    def __init__(self, action, state, delta=0.25, epsilon=0.1, sample=False):
        self.delta = delta
        self.epsilon = epsilon
        self.A_ = int((action-1) / 2)
        self.k = int(state / 2)
        self.depth = math.ceil(math.log((self.A_-1)*self.k + 1, self.A_) - 1)

        self.state = None
        self.initial_state = int(2 * (self.A_**self.depth-1) / (self.A_-1))
        self.good_state = None

        self.action_space = spaces.Discrete(2*self.A_ + 1)
        self.observation_space = spaces.Discrete(2 * self.k)

        if sample:
            self.delta, self.epsilon = self.Direichelt(np.array([self.delta, self.epsilon, 1-self.delta-self.epsilon]))

    def step(self, action):
        """
        Take an action and move to next state

        :param action:
        :return:
        """
        random_unif = np.random.uniform()
        reward = 0
        # Type-1 state
        if self.state % 2 == 1:
            if random_unif > self.delta:  # stay in type-1 state
                reward = 1
            else:  # move to type-0 state
                self.state -= 1
        # Type-0 state
        else:
            if action == 0:
                next_state = int((self.state/2 - 1) / self.A_) * 2
                self.state = max(next_state, 0)  # back to the root
            elif action <= self.A_:
                # the action 1 is the good action for the good state
                if self.state == self.good_state and action == 1:
                    trans_prob = self.delta + self.epsilon
                else:
                    trans_prob = self.delta
                if random_unif > trans_prob:  # stay in type-0 state
                    pass
                else:  # move to type-1 state
                    self.state += 1
                    reward = 1
            else:
                next_state = int((self.A_ * (self.state/2 - 1) + action)) * 2
                self.state = next_state if next_state < 2 * self.k else self.state  # go to a leaf

        return self.state, reward, False, {}

    def reset(self):
        self.state = self.initial_state
        if self.depth == 1:
            self.good_state = (self.k-1) * 2
        else:
            self.good_state = (self.k-1) * 2 if (self.k-1) * 2 >= self.state + self.A_ * 2 else self.state - 2
        return self.state

    def render(self, mode='human'):
        return None

    def close(self):
        return None

    def A(self):
        # Get the action space size
        return 2 * self.A_ + 1

    def S(self):
        # Get the state space size
        return 2 * self.k

    def bestR(self):
        # Get the best average rewards
        return (self.delta+self.epsilon) / (2*self.delta+self.epsilon)

    def BackwardPolicy(self, state):
        """
        Compute the policy to go backward to the root state

        :param state: Current Type-0 state
        :return:
            root_state: The rood state of current one
            action: the action it take from root to current
        """
        if state == 0:
            return -1, -1
        else:
            root_state = int((state/2 - 1) / self.A_) * 2
            action = int((state/2 - 1) % self.A_ + self.A_ + 1)
            return root_state, action

    def Direichelt(self, M_vec):

        # Sample from dirichlet distribution
        # If there is zero exists, skip that zero
        assert np.all(M_vec >= 0), "There is negative para in Dirichlet Distribution"
        result = np.zeros(len(M_vec))
        i = 0
        while True:
            result = np.random.dirichlet(M_vec)
            i+=1
            if result[0] + result[1] < 2/3 and result[0] > result[1]:
                print("sample time", i)
                break
            if i > 20:
                print("Sample too much time")
        return result[0], result[1]
