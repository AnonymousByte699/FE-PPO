import torch
import matplotlib.pyplot as plt
from ppo_continuous import PPO_continuous, ReplayBuffer
import numpy as np
from AD_hospital_env_only_drug import ADEnv


def env_run(rate):
    # avail, worsen, static, dynamic
    s_rand = torch.zeros((4, 50))
    s_hos = torch.zeros((4, 50))
    s_mean = torch.zeros((4, 50))
    s_u = torch.zeros((4, 50))
    s_f = torch.zeros((4, 50))
    s_feppo = torch.zeros((4, 50))

    print("----------env:rate=", rate, "----------")

    AD_env = ADEnv(rate=rate)
    hospital_num = AD_env.doctor_num.shape[0]

    time = 20

    for m in range(50):
        AD_env.reset()
        dy_fair = torch.zeros((20, 1))
        for t in range(time):
            rand_seq = torch.rand(hospital_num)
            distribution = rand_seq / torch.sum(rand_seq)
            AD_env.step_none(distribution)
            dy_fair[t] = torch.var(AD_env.get_region_avail()) * 1000
        s_rand[0, m] = (AD_env.get_avail_rate())
        s_rand[1, m] = (AD_env.get_worsen_rate())
        s_rand[2, m] = (torch.var(AD_env.get_region_avail()))
        s_rand[3, m] = (torch.mean(dy_fair))
    print("rand:")
    print("avail: ", torch.mean(s_rand[0, :]), "+", torch.std(s_rand[0, :]))
    print("worsen: ", torch.mean(s_rand[1, :]), "+", torch.std(s_rand[1, :]))
    print("static: ", torch.mean(s_rand[2, :]), "+", torch.std(s_rand[2, :]))
    print("dynamic: ", torch.mean(s_rand[3, :]), "+", torch.std(s_rand[3, :]))

    for m in range(50):
        AD_env.reset()
        dy_fair = torch.zeros((20, 1))
        for t in range(time):
            distribution = AD_env.doctor_num / torch.sum(AD_env.doctor_num, dim=0)
            AD_env.step_none(distribution)
            dy_fair[t] = torch.var(AD_env.get_region_avail()) * 1000
        s_hos[0, m] = (AD_env.get_avail_rate())
        s_hos[1, m] = (AD_env.get_worsen_rate())
        s_hos[2, m] = (torch.var(AD_env.get_region_avail()))
        s_hos[3, m] = (torch.mean(dy_fair))
    print("hos:")
    print("avail: ", torch.mean(s_hos[0, :]), "+", torch.std(s_hos[0, :]))
    print("worsen: ", torch.mean(s_hos[1, :]), "+", torch.std(s_hos[1, :]))
    print("static: ", torch.mean(s_hos[2, :]), "+", torch.std(s_hos[2, :]))
    print("dynamic: ", torch.mean(s_hos[3, :]), "+", torch.std(s_hos[3, :]))

    for m in range(50):
        AD_env.reset()
        dy_fair = torch.zeros((20, 1))
        for t in range(time):
            distribution = torch.ones(hospital_num) * (1 / hospital_num)
            AD_env.step_none(distribution)
            dy_fair[t] = torch.var(AD_env.get_region_avail()) * 1000
        s_mean[0, m] = (AD_env.get_avail_rate())
        s_mean[1, m] = (AD_env.get_worsen_rate())
        s_mean[2, m] = (torch.var(AD_env.get_region_avail()))
        s_mean[3, m] = (torch.mean(dy_fair))
    print("mean:")
    print("avail: ", torch.mean(s_mean[0, :]), "+", torch.std(s_mean[0, :]))
    print("worsen: ", torch.mean(s_mean[1, :]), "+", torch.std(s_mean[1, :]))
    print("static: ", torch.mean(s_mean[2, :]), "+", torch.std(s_mean[2, :]))
    print("dynamic: ", torch.mean(s_mean[3, :]), "+", torch.std(s_mean[3, :]))

    AD_env.reset()
    if rate == 2:
        agent = torch.load('./logs/utility_baseline')
    elif rate == 1.7:
        agent = torch.load('./logs/env17_u')
    elif rate == 2.3:
        agent = torch.load('./logs/env23_u')
    with torch.no_grad():
        for m in range(50):
            AD_env.reset()
            dy_fair = torch.zeros((20, 1))
            for t in range(time):
                state = AD_env.get_obs()
                a, _ = agent.choose_action(state)
                AD_env.step(a)
                dy_fair[t] = torch.var(AD_env.get_region_avail()) * 1000
            s_u[0, m] = (AD_env.get_avail_rate())
            s_u[1, m] = (AD_env.get_worsen_rate())
            s_u[2, m] = (torch.var(AD_env.get_region_avail()))
            s_u[3, m] = (torch.mean(dy_fair))
    print("u:")
    print("avail: ", torch.mean(s_u[0, :]), "+", torch.std(s_u[0, :]))
    print("worsen: ", torch.mean(s_u[1, :]), "+", torch.std(s_u[1, :]))
    print("static: ", torch.mean(s_u[2, :]), "+", torch.std(s_u[2, :]))
    print("dynamic: ", torch.mean(s_u[3, :]), "+", torch.std(s_u[3, :]))

    agent = torch.load('./logs/only_fair_100')
    with torch.no_grad():
        for m in range(50):
            AD_env.reset()
            dy_fair = torch.zeros((20, 1))
            for t in range(time):
                state = AD_env.get_obs()
                a, _ = agent.choose_action(state)
                AD_env.step(a)
                dy_fair[t] = torch.var(AD_env.get_region_avail()) * 1000
            s_f[0, m] = (AD_env.get_avail_rate())
            s_f[1, m] = (AD_env.get_worsen_rate())
            s_f[2, m] = (torch.var(AD_env.get_region_avail()))
            s_f[3, m] = (torch.mean(dy_fair))
    print("f:")
    print("avail: ", torch.mean(s_f[0, :]), "+", torch.std(s_f[0, :]))
    print("worsen: ", torch.mean(s_f[1, :]), "+", torch.std(s_f[1, :]))
    print("static: ", torch.mean(s_f[2, :]), "+", torch.std(s_f[2, :]))
    print("dynamic: ", torch.mean(s_f[3, :]), "+", torch.std(s_f[3, :]))

    if rate == 2:
        agent = torch.load('./logs/divide_110')
    elif rate == 1.7:
        agent = torch.load('./logs/env17')
    elif rate == 2.3:
        agent = torch.load('./logs/env23')
    with torch.no_grad():
        for m in range(50):
            AD_env.reset()
            dy_fair = torch.zeros((20, 1))
            for t in range(time):
                state = AD_env.get_obs()
                a, _ = agent.choose_action(state)
                AD_env.step(a)
                dy_fair[t] = torch.var(AD_env.get_region_avail()) * 1000
            s_feppo[0, m] = (AD_env.get_avail_rate())
            s_feppo[1, m] = (AD_env.get_worsen_rate())
            s_feppo[2, m] = (torch.var(AD_env.get_region_avail()))
            s_feppo[3, m] = (torch.mean(dy_fair))
    print("feppo:")
    print("avail: ", torch.mean(s_feppo[0, :]), "+", torch.std(s_feppo[0, :]))
    print("worsen: ", torch.mean(s_feppo[1, :]), "+", torch.std(s_feppo[1, :]))
    print("static: ", torch.mean(s_feppo[2, :]), "+", torch.std(s_feppo[2, :]))
    print("dynamic: ", torch.mean(s_feppo[3, :]), "+", torch.std(s_feppo[3, :]))

env_run(2)
env_run(1.7)
env_run(2.3)