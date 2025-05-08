import torch
from ppo_continuous import PPO_continuous, ReplayBuffer
import numpy as np
from AD_hospital_env_fair import ADEnv
from normalization import Normalization, RewardScaling
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default=2, type=float, help="env drug allocation rate")
    parser.add_argument('--step', default=20, type=int, help="steps in an episode")
    parser.add_argument('--episode', default=20000, type=int, help="episodes in training")
    parser.add_argument('--batch', default=2048, type=int, help="batch size in replay buffer")
    parser.add_argument('--model_name', default='env2', type=str, help="model name to save")

    args = parser.parse_args()

    env = ADEnv(args.env)
    env.reset()

    hospital_num = env.doctor_num.shape[0]

    num_features = env.get_obs().shape[0]
    n_actions = env.doctor_num.shape[0]

    n_episode = args.step
    max_step = args.episode * n_episode

    evaluate_rewards = []  # record the rewards during the evaluating
    total_steps = 0  # record the total steps during the training

    reward_norm = Normalization(shape=1)

    replay_buffer = ReplayBuffer(batch_size=2048, num_features=num_features, action_dim=hospital_num)

    agent = PPO_continuous(num_features=num_features, hidden_size=256, num_actions=n_actions, max_step=max_step)

    # # build a tensorboard
    # writer = SummaryWriter(log_dir='./log')

    max_reward = 0
    while total_steps < max_step:
        env.reset()

        # distribution = torch.ones(hospital_num) * (1 / hospital_num)
        # env.step(distribution)
        # env.drug_num = torch.zeros(env.doctor_num.shape[0])

        state = env.get_obs()
        episode_steps = 0
        # in job env, dw is False, done is True only depends on episode_step==n_episode
        done = False
        dw = False

        evaluate_reward = 0
        while episode_steps < n_episode:

            episode_steps += 1
            a, a_logprob = agent.choose_action(state)

            reward = env.step(a)

            evaluate_reward += reward

            # reward = reward_norm(reward)
            state_next = env.get_obs()

            if episode_steps == n_episode:
                done = True
            else:
                done = False
            replay_buffer.store(state, a, a_logprob, reward, state_next, dw, done)

            state = state_next
            total_steps += 1
            # print(total_steps)

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == replay_buffer.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # # Evaluate the policy every 'evaluate_freq' steps
            # if total_steps % evaluate_freq == 0:
            #     evaluate_num += 1
            #     evaluate_reward = evaluate_policy(env_evaluate, agent)
            #     evaluate_rewards.append(evaluate_reward)
            #     print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
            #     # writer.add_scalar('step_rewards', evaluate_rewards[-1], global_step=total_steps)
        evaluate_rewards.append(evaluate_reward)
        print("step_:{} \t evaluate_reward:{} \t".format(total_steps, evaluate_reward))
        # print(torch.max(env.get_region_avail()) - torch.min(env.get_region_avail()))

    torch.save(agent, args.model_name)
    np.savez(args.model_name, evaluate_rewards)


if __name__ == "__main__":
    main()
