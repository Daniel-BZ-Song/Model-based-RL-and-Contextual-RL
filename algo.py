from env import *
from scipy.stats import norm


class ModelBaseRL(object):
    """
    Model based reinforcement learning algorithm

    Provide general properties used in different algorithm
    """

    def __init__(self, env, verbose=False):
        self.env = env
        self.A = env.A()
        self.S = env.S()
        self.verbose = verbose

        self.N_sa = np.zeros((self.S, self.A))
        self.N_sas = np.zeros((self.S, self.A, self.S))
        self.R_sa = np.zeros((self.S, self.A))

        # value function and policy
        self.policy = np.zeros(self.S)
        self.value = np.zeros(self.S)

        # Regret
        self.reward_ls = []
        self.reward_path_cum = None
        self.reward_path_avg = None
        self.regret_path_cum = None
        self.regret_path_avg = None

    def get_regret_path(self):
        T = len(self.reward_ls)
        self.reward_path_cum = np.cumsum(self.reward_ls)
        self.regret_path_cum = (np.arange(T)+1)*self.env.bestR() - self.reward_path_cum
        self.regret_path_avg = self.regret_path_cum / (np.arange(T)+1)
        self.reward_path_avg = self.reward_path_cum / (np.arange(T) + 1)
        return 0


class Optimal_tree(ModelBaseRL):

    def __init__(self, env, verbose=False):
        super().__init__(env, verbose)

    def learn(self, T):
        t = 1
        s_t = self.env.reset()
        self.policy = np.zeros(self.S)
        # Find optimal path
        good_state = self.env.good_state
        self.policy[good_state] = 1

        path, action = self.env.BackwardPolicy(good_state)
        while path >= 0:
            self.policy[path] = action
            path, action = self.env.BackwardPolicy(path)

        # Start learning
        while t <= T:
            a_t = self.policy[s_t]
            s_new, r_t, _, _ = self.env.step(a_t)

            s_t = s_new
            self.reward_ls.append(r_t)
            t += 1


class Optimal_river(ModelBaseRL):
    def __init__(self, env, verbose=False):
        super().__init__(env, verbose)

    def learn(self, T):
        t = 1
        s_t = self.env.reset()
        self.policy = np.ones(self.S).astype(int)
        while t <= T:
            a_t = self.policy[s_t]
            s_new, r_t, _, _ = self.env.step(a_t)

            self.R_sa[s_t, a_t] += r_t
            self.N_sa[s_t, a_t] += 1
            self.N_sas[s_t, a_t, s_new] += 1
            s_t = s_new
            self.reward_ls.append(r_t)
            t += 1


class UCRL2(ModelBaseRL):
    """
    UCRL2 Algorithm

    Described in Jaksch et al. [2010], Part-3 The Lower Bound UCRL2 Algorithm
    """
    def __init__(self, env, delta=0.2, v_iter_epsilon=0.01, v_iter_max_iter=10000, verbose=False):
        super().__init__(env, verbose)
        self.V_sa = np.zeros((self.S, self.A))
        self.t_k = 1
        self.episode = 0
        # estimate mdp
        self.r_sa = np.zeros((self.S, self.A))
        self.P_sas = np.zeros((self.S, self.A, self.S))
        self.delta = delta

        # UCB part
        self.p_bound = np.zeros((self.S, self.A))
        self.r_bound = np.zeros((self.S, self.A))
        self.max_P_sas = np.zeros((self.S, self.A, self.S))

        # Value iteration
        self.v_iter_epsilon = v_iter_epsilon
        self.v_iter_max_iter = v_iter_max_iter

    def learn(self, T):
        t = 1
        s_t = self.env.reset()

        while t <= T:
            if self.verbose and self.episode % 100 == 1:
                print("Episode: %d; Time: %d" % (self.episode, t))

            # Initialized episode k
            self.episode += 1
            self.t_k = t
            self.N_sa += self.V_sa
            self.V_sa = np.zeros(self.V_sa.shape)
            N_sa_3d = np.repeat(self.N_sa[:, :, np.newaxis], self.S, axis=2)
            self.P_sas = self.N_sas / np.maximum(1, N_sa_3d)

            # Compute policy
            tmp_fac_r = np.log(2*self.S*self.A*self.t_k/self.delta)
            tmp_fac_p = self.S * np.log(2*self.A*self.t_k/self.delta)
            self.r_bound = np.sqrt(7 / 2 * tmp_fac_r / np.maximum(1, self.N_sa))
            self.p_bound = np.sqrt(14 * tmp_fac_p / np.maximum(1, self.N_sa))
            self.r_sa = self.R_sa / np.maximum(1, self.N_sa) + self.r_bound

            # Extended value iteration
            self.ExtendedValueIter()

            # Execute policy
            a_t = self.policy[s_t]
            while self.V_sa[s_t, a_t] < max(1, self.N_sa[s_t, a_t]) and t <= T:
                s_new, r_t, _, _ = self.env.step(a_t)
                self.R_sa[s_t, a_t] += r_t
                self.V_sa[s_t, a_t] += 1
                self.N_sas[s_t, a_t, s_new] += 1
                s_t = s_new
                a_t = self.policy[s_t]
                self.reward_ls.append(r_t)
                t += 1

    def ExtendedValueIter(self):
        # Find the optimal policy by extended value iteration
        count = 0

        while True:
            count += 1
            self.InnerMaxP()
            value_prev = self.value.copy()

            # Bellman Operator: compute policy and value functions
            # modify from mdptoolbox library
            self.policy, self.value = self.BellmanOperator()

            if max(self.value - value_prev) - min(self.value - value_prev)< 1/np.sqrt(self.t_k) or count > self.v_iter_max_iter:
                break

    def InnerMaxP(self):
        # Compute inner maximum in extended value iteration
        sort_idx = (-self.value).argsort()
        self.max_P_sas = self.P_sas.copy()
        self.max_P_sas[:, :, sort_idx[0]] = np.minimum(1, self.P_sas[:, :, sort_idx[0]] + self.p_bound/2)
        j = self.S - 1  # l in original paper
        while np.any(np.sum(self.max_P_sas, axis=2) > 1):
            tmp_idx = np.sum(self.max_P_sas, axis=2) > 1
            # print(tmp_idx.shape)
            # print(np.maximum(0, 1 - np.sum(np.delete(self.max_P_sas, sort_idx[j], axis=2), axis=2)).shape)
            # print(self.max_P_sas[tmp_idx, sort_idx[j]].shape)
            self.max_P_sas[tmp_idx, sort_idx[j]] = \
                np.maximum(0, 1 - np.sum(np.delete(self.max_P_sas, sort_idx[j], axis=2), axis=2))[tmp_idx]
            j -= 1

    def BellmanOperator(self):
        # Iterate through all Q function

        Q = np.zeros((self.A, self.S))
        for a in range(self.A):
            Q[a] = self.r_sa[:, a] + self.max_P_sas[:, a, :].dot(self.value)

        return Q.argmax(axis=0), Q.max(axis=0)


class PSRL(ModelBaseRL):
    """
    Posterior Sampling Reinforcement Learning

    Described in Shipra Agrawal and Randy Jia [2017]
    """

    def __init__(self, env, rho=0.2, v_iter_epsilon=0.01, multi_sample=True, v_iter_max_iter=10000, prior=False, verbose=False):
        super().__init__(env, verbose)

        self.episode = 0
        self.rho = rho
        self.V_sa = np.zeros((self.S, self.A))
        self.t_k = 1
        self.no_converge = 0
        # estimate mdp

        self.r_sa = None
        self.P_sas = np.zeros((self.S, self.A, self.S))

        # Posterior Sampling
        self.prior = prior
        if multi_sample:
            self.phi = self.S
        else:
            self.phi = 1
        self.omega = None
        self.kai = None
        self.eta = None
        self.M_sas = np.ones((self.S, self.A, self.S))
        self.Q_sasp = None  # size: phi * S * A * S

        # Reward
        self.r_sa_ls = [[[] for i in range(self.A)] for k in range(self.S)]


        # Value iteration
        self.v_iter_epsilon = v_iter_epsilon
        self.v_iter_max_iter = v_iter_max_iter

    def learn(self, T):
        # Initialize the parameter
        C = 7 ** 2  # (32/((1-norm.cdf(0.5))/2)**4)
        # self.phi = self.S  #self.S #int(C * self.S * np.log(self.S*self.A/self.rho))
        self.kai = 2 * np.log(T/self.rho)
        self.omega = np.log(T/self.rho)
        # if self.prior:
        #     self.omega = self.omega * self.env.get_transition() +1
        self.eta = 1  #np.sqrt(T*self.S/self.A) + 12*self.omega*self.S**4
        self.Q_sasp = np.zeros((self.S, self.A, self.S, self.phi))
        self.r_sa = np.zeros((self.S, self.A, self.phi))
        t = 1
        s_t = self.env.reset()
        while t <= T:
            if self.verbose and self.episode % 100 == 1:
                print("Episode: %d; Time: %d" % (self.episode, t))

            # Initialized episode k
            self.episode += 1
            self.t_k = t
            self.N_sa += self.V_sa
            self.M_sas = (self.N_sas+self.omega) / self.kai
            self.V_sa = np.zeros(self.V_sa.shape)
            N_sa_3d = np.repeat(self.N_sa[:, :, np.newaxis], self.S, axis=2)
            self.P_sas = self.N_sas / np.maximum(1, N_sa_3d)
            #self.r_sa = self.R_sa / np.maximum(1, self.N_sa)  # Here we use the sample reward directly

            # Sample Transition Probability
            self.PostSampling()
            self.RewardSampling()
            # Extended value iteration
            self.ExtendedValueIter()

            # Execute policy
            while True:
                t += 1
                a_t = self.policy[s_t]
                s_new, r_t, _, _ = self.env.step(a_t)
                self.R_sa[s_t, a_t] += r_t
                self.V_sa[s_t, a_t] += 1
                self.N_sas[s_t, a_t, s_new] += 1
                self.reward_ls.append(r_t)
                self.r_sa_ls[s_t][a_t].append(r_t)
                if self.V_sa[s_t, a_t] >= self.N_sa[s_t, a_t]:
                    s_t = s_new
                    break
                s_t = s_new
                if t > T:
                    print("PSRL Done")
                    break

    def Dirichlet(self, M_vec):
        """
        Sample from Dirichlet distribution

        Apply to np.apply_along_axis in PostSampling()

        :param M_vec: 1-d array, S,
        :return: sample, size: S * phi
        """
        return np.random.dirichlet(M_vec, self.phi).T

    def NormalGamma(self, para_vec):
        mu, lbd, alpha, beta = para_vec
        tao = np.random.gamma(alpha, 1/beta, self.phi)
        sig_2 = 1 / (tao*lbd)
        x = np.random.normal(1, 1, self.phi)
        return (mu + x * np.sqrt(sig_2)).T

    def get_std(self, ls):
        n = len(ls)
        if n <= 1:
            return 0
        else:
            return np.std(ls)**2

    def RewardSampling(self):
        r_sa_para = np.ones((self.S, self.A, 4))
        r_sa_para[:, :, 0] = (1+self.R_sa) / (1+self.N_sa)
        r_sa_para[:, :, 1] = 1 + self.N_sa
        r_sa_para[:, :, 2] = 1 + self.N_sa / 2
        r_sa_std = np.array([[self.get_std(ls) for ls in LS] for LS in self.r_sa_ls])
        #self.para = r_sa_std
        r_sa_para[:, :, 3] = 1 + (self.N_sa*r_sa_std + (self.N_sa*(self.R_sa/np.maximum(1,self.N_sa)-1)**2)/(1+self.N_sa)) / 2
        self.para = r_sa_para
        self.r_sa = np.apply_along_axis(self.NormalGamma, 2, r_sa_para)

    def SimpleSampling(self, N_vec):
        """

        :param N_vec:
        :return: Q_simple, size: S * phi
        """
        assert len(N_vec) == self.S, "The shape of N_vec is wrong"
        N_sum = np.sum(N_vec)
        P = N_vec / np.maximum(1, N_sum)
        delta_tmp = np.sqrt(3*P*np.log(4*self.S)/np.maximum(1, N_sum)) + 3*np.log(4*self.S)/np.maximum(1, N_sum)
        delta = np.minimum(P, delta_tmp)
        assert delta.shape == N_vec.shape, "The shape of delta is wrong"
        P_ = P - delta

        z = np.eye(self.S)[np.random.randint(self.S, size=self.phi)].T
        Q_simple = np.repeat(P_[:, np.newaxis], self.phi, axis=1)
        Q_simple += (1 - np.sum(P_)) * z
        # assert np.sum(Q_simple,axis=0) == np.ones(self.phi)
        return Q_simple

    def PostSampling(self):
        tmp_idx = self.N_sa < self.eta
        self.Q_sasp = np.apply_along_axis(self.Dirichlet, 2, self.M_sas)
        if np.sum(tmp_idx) > 0:
            self.Q_sasp[tmp_idx] = np.apply_along_axis(self.SimpleSampling, 1, self.N_sas[tmp_idx])

    def ExtendedValueIter(self):
        # Find the optimal policy by extended value iteration
        count = 0

        while True:
            count += 1
            value_prev = self.value.copy()

            # Bellman Operator: compute policy and value functions
            # modify from mdptoolbox library
            self.policy, self.value = self.BellmanOperator()

            # if max(self.value - value_prev) - min(self.value - value_prev) < 1/np.sqrt(self.t_k) :
            if max(self.value - value_prev) < self.v_iter_epsilon:
                break
            if count > self.v_iter_max_iter:
                # print("not converge")
                self.no_converge += 1
                # print(max(self.value - value_prev), min(self.value - value_prev))
                break

    def BellmanOperator(self):
        # Iterate through all Q function
        Q = np.zeros((self.phi*self.A, self.S))
        for aa in range(self.phi*self.A):
            a = aa % self.A
            q = aa // self.A
            Q[a] = self.r_sa[:, a, q] + self.Q_sasp[:, a, :, q].dot(self.value)
        return Q.argmax(axis=0) % self.A, Q.max(axis=0)


class UCBVI(ModelBaseRL):
    """
    UCBVI Algorithm

    Described in Gheshlaghi Azar et al., 2017, Part-3 Upper confidence bound value iteration
    """
    def __init__(self, env, delta=0.2, v_iter_epsilon=0.01, v_iter_max_iter=10000, verbose=False):
        super().__init__(env, verbose)
        self.V_sa = np.zeros((self.S, self.A))
        self.t_k = 1
        self.episode = 0
        # estimate mdp
        self.r_sa = np.zeros((self.S, self.A))
        self.P_sas = np.zeros((self.S, self.A, self.S))
        self.delta = delta

        # UCB part
        self.p_bound = np.zeros((self.S, self.A))
        self.r_bound = np.zeros((self.S, self.A))
        self.max_P_sas = np.zeros((self.S, self.A, self.S))

        # Value iteration
        self.v_iter_epsilon = v_iter_epsilon
        self.v_iter_max_iter = v_iter_max_iter

    def learn(self, T):
        t = 1
        s_t = self.env.reset()

        while t <= T:
            if self.verbose and self.episode % 100 == 1:
                print("Episode: %d; Time: %d" % (self.episode, t))

            # Initialized episode k
            self.episode += 1
            self.t_k = t
            self.N_sa += self.V_sa
            self.V_sa = np.zeros(self.V_sa.shape)
            N_sa_3d = np.repeat(self.N_sa[:, :, np.newaxis], self.S, axis=2)
            self.P_sas = self.N_sas / np.maximum(1, N_sa_3d)

            # Compute policy
            tmp_fac_r = np.log(2*self.S*self.A*self.t_k/self.delta)
            tmp_fac_p = self.S * np.log(2*self.A*self.t_k/self.delta)
            self.r_bound = np.sqrt(7 / 2 * tmp_fac_r / np.maximum(1, self.N_sa))
            self.p_bound = np.sqrt(14 * tmp_fac_p / np.maximum(1, self.N_sa))
            self.r_sa = self.R_sa / np.maximum(1, self.N_sa) + self.r_bound

            # Extended value iteration
            self.ExtendedValueIter()

            # Execute policy
            a_t = self.policy[s_t]
            while self.V_sa[s_t, a_t] < max(1, self.N_sa[s_t, a_t]) and t <= T:
                s_new, r_t, _, _ = self.env.step(a_t)
                self.R_sa[s_t, a_t] += r_t
                self.V_sa[s_t, a_t] += 1
                self.N_sas[s_t, a_t, s_new] += 1
                s_t = s_new
                a_t = self.policy[s_t]
                self.reward_ls.append(r_t)
                t += 1


