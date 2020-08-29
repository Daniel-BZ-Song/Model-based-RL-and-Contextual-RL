from algo import UCRL2, PSRL, Optimal_river, Optimal_tree
from env import RiverSwim, TreeMDP
import matplotlib.pyplot as plt
import time
import pickle
import numpy as np
from multiprocessing import Pool, cpu_count
import os

ACTION = 0
STATE = 6
ENV_TYPE = "River"
TOTAL_TIME = 5000
BAYESIAN = False


def test_case(ii):
    epic_start = time.time()
    np.random.seed()
    print("Round %d start:" % (ii + 1))
    if ENV_TYPE == "River":
        test_env = RiverSwim(STATE, sample=BAYESIAN)
        best = Optimal_river(test_env, verbose=True)
    elif ENV_TYPE == "Tree":
        test_env = TreeMDP(action=ACTION, state=STATE, delta=0.3, epsilon=0.1, sample=BAYESIAN)
        best = Optimal_tree(test_env, verbose=True)
    best.learn(TOTAL_TIME)
    ps_single = PSRL(test_env, multi_sample=False, prior=BAYESIAN, verbose=True)
    ps_single.learn(TOTAL_TIME)
    ps_multi = PSRL(test_env, multi_sample=True, verbose=True)
    ps_multi.learn(TOTAL_TIME)
    uc = UCRL2(test_env, delta=0.05, verbose=True)
    uc.learn(TOTAL_TIME)

    best.get_regret_path()
    ps_multi.get_regret_path()
    ps_single.get_regret_path()
    uc.get_regret_path()
    # if best.reward_path_cum[-1] - ps.reward_path_cum[-1] > 20000:
    #     bug = ps
    # plt.plot((best.reward_path_cum - ps.reward_path_cum), color="firebrick", alpha=0.5)
    # plt.plot((best.reward_path_cum - uc.reward_path_cum), color="royalblue", alpha=0.5)
    print("Round %d end; Total time: %.2f seconds\n" % (ii+1, time.time() - epic_start))
    best_reward = (np.arange(TOTAL_TIME) + 1) * best.reward_path_avg[-1]
    ps_regret_single = (best_reward - ps_single.reward_path_cum)
    ps_regret_multi = (best_reward - ps_multi.reward_path_cum)
    uc_regret = (best_reward - uc.reward_path_cum)
    return ps_regret_single[::1000], ps_regret_multi[::1000], uc_regret[::1000]


def plot_regret(PS_regret_single, PS_regret_multi, UC_regret):
    ps_s = np.array(PS_regret_single).T  # n*path
    ps_m = np.array(PS_regret_multi).T
    uc = np.array(UC_regret).T

    num_plot = 3
    idx = ps_s[-1, :].argsort()[::-1][:num_plot]
    cm = plt.get_cmap('inferno')
    fig, ax = plt.subplots(1, figsize=(8, 6))
    color_ls = [cm(1. * (i + 0.5) / num_plot) for i in range(num_plot)]
    color_ls_sub = [cm(1. * i / num_plot) for i in range(num_plot)]
    for i in range(num_plot):
        ax.plot(ps_s[:, idx[i]], color=color_ls[i], alpha=0.7, label="Path %d_PS" % (i + 1))
        ax.plot(uc[:, idx[i]], color=color_ls_sub[i], alpha=0.7, label="Path %d_UC" % (i + 1))
    fig.legend()
    fig.suptitle("Top %d Path Analysis\n%s: %d actions %s states - %s" % (num_plot, ENV_TYPE, ACTION, STATE, ("Bayesian" if BAYESIAN else "NonBayesian")),y=1, fontsize=13)
    fig.savefig('%s_A%d_S%d-%s_top path.png' % (ENV_TYPE, ACTION, STATE, ("Bayesian" if BAYESIAN else "NonBayesian")))
    fig.show()

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    ax[0][0].plot(ps_m.mean(axis=1), color="firebrick", alpha=0.5, label="PSRL_multi")
    ax[0][0].plot(uc.mean(axis=1), color="royalblue", alpha=0.5, label="UCRL2")
    ax[0][0].set_title("Expected Regret for UCRL2 and PSRL")
    ax[0][0].legend()

    ax[0][1].plot(ps_m.mean(axis=1), color="firebrick", alpha=0.5, label="PSRL_multi")
    ax[0][1].plot(ps_s.mean(axis=1), color="darkorange", alpha=0.5, label="PSRL_single")
    ax[0][1].set_title("Expected Regret for PSRL with single & multi sample")
    ax[0][1].legend()

    ax[1][0].plot(ps_m, color="firebrick", alpha=0.4)
    ax[1][0].plot(uc, color="royalblue", alpha=0.4)
    ax[1][0].set_title("32 paths for UCRL2 and PSRL")

    ax[1][1].plot(ps_m, color="firebrick", alpha=0.4)
    ax[1][1].plot(ps_s, color="darkorange", alpha=0.4)
    ax[1][1].set_title("32 paths for PSRL with single & multi sample")

    fig.suptitle("Model Results Summary\n%s: %d actions %s states - %s" % (ENV_TYPE, ACTION, STATE, ("Bayesian" if BAYESIAN else "NonBayesian")), y=0.99, fontsize=13)
    fig.savefig('%s_A%d_S%d-%s.png' % (ENV_TYPE, ACTION, STATE, ("Bayesian" if BAYESIAN else "NonBayesian")))
    fig.show()


if __name__ == "__main__":
    print("CPU内核数:{}".format(cpu_count()))
    print('当前母进程: {}'.format(os.getpid()))
    time_start = time.time()
    para_ls = [("River", 2, 48, 1e8, True)]
        # ("Tree", 5, 30, 1e8, True)]
              # ("River", 2, 36, 1e7, True),

    for para in para_ls:
        ENV_TYPE, ACTION, STATE, TOTAL_TIME, BAYESIAN = para
        print(para)
        p = Pool(32)
        PS_regret_single, PS_regret_multi, UC_regret = zip(*p.map(test_case, np.arange(32)))
        print('等待所有子进程完成')
        p.close()
        p.join()
        with open('path.pkl', 'wb') as output:  # Overwrites any existing file.
            pickle.dump((PS_regret_single, PS_regret_multi, UC_regret), output)
        plot_regret(PS_regret_single, PS_regret_multi, UC_regret)
    print("Total time: %.2f seconds" % (time.time() - time_start))





