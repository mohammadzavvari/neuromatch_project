import argparse
import datetime
import os
from time import time
import z_experiments
import numpy as np
import torch
from matplotlib import pyplot as plt
from multiagent import scenarios
from multiagent.environment import MultiAgentEnv

from MADDPG import MADDPG

# We now name our folderbased on arguments if loading goty edition
# We also add a new scenario to simple_tag_goty_edition.py



# EXPERIMENT ARGS GLOBALS
DEFAULT_ARGS = {'env': 'simple_tag',
                'episode_length': 25, 
                'episode_num': 30000, 
                'gamma': 0.95, 
                'buffer_capacity': int(1e6), 
                'batch_size': 1024, 
                'actor_lr': 1e-2, 
                'critic_lr': 1e-2, 
                'steps_before_learn': 5e4, 
                'learn_interval': 100, 
                'save_interval': 100, 
                'tau': 0.02}

# Old batch runner 
# Just pass these as the args to the runner, and it will run the experiment
# you can also specify any changes to the any of the key, value pairs
# in DEFAULT_ARGS global to run it
# this will run as normal, putting results into the folder named with env
no_adv_sharing_args = {'env': 'simple_tag_no_adv_sharing'}
big_bounds_args = {'env': 'simple_tag_big_bounds'}

# SHINY NEW BATCH RUNNER
# We can change the parameters of the new scenario directly
#   env = 'simple_tag_goty_edition'

# We can use any of the the world args as kwargs to make_world to set 
# parameters of simple tag game of the year edition

default_world_args = {
    'num_good_agents': 4,
    'num_adversaries': 2,
    'num_landmarks': 2,
    'out_of_bound_punishment': 10,
    'hard_boundary': False,
    'old_collaborative': False,
    'good_collaborative': False,
    'bad_collaborative': False,
    'remove_old_adv_sharing': True,
    'shape_reward': False}

# Example
# We need to pass both of these to use the goty edition and world args
example_main_args = {'env' : 'simple_tag_goty_edition',}
example_world_args = {
    'out_of_bound_punishment': 100,
    'hard_boundary': True,
    'good_collaborative': True,
    'bad_collaborative': False}


# world.collaborative = True # old colaborative method
# ind_adversary = True # remove old adversary reward sharing method
# num_good_agents = 4
# num_adversaries = 2
# good_collaborative = True
# bad_collaborative = False


def arg_parser(args_dict) -> argparse.Namespace:
    "convert arg dict to argparse object"
    parser = argparse.ArgumentParser()
    for key, value in args_dict.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    return parser.parse_args()

def get_args(new_args, default_args=DEFAULT_ARGS) -> argparse.Namespace:
    "return default args with any specified arguments updates as argparse object"
    args_dict = default_args
    for key, value in new_args.items():
        args_dict[key] = value
    return arg_parser(args_dict)

def get_world_args(new_args, default_args) -> dict:
    "return default world args with any specified arguments updates as dict"
    # I should just roll this into get_args but I'm lazy right now sorry :(
    world_args = default_args
    for key, value in new_args.items():
        world_args[key] = value
    return world_args

def default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default = 'simple_tag', help='name of the environment',
                        choices=['simple_adversary', 'simple_crypto', 'simple_push', 'simple_reference',
                                 'simple_speaker_listener', 'simple_spread', 'simple_tag',
                                 'simple_world_comm', 'simple_tag_4_4'])
    parser.add_argument('--episode-length', type=int, default=25, help='steps per episode')
    parser.add_argument('--episode-num', type=int, default=30000, help='total number of episode')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer-capacity', default=int(1e6))
    parser.add_argument('--batch-size', default=1024)
    parser.add_argument('--actor-lr', type=float, default=1e-2, help='learning rate of actor')
    parser.add_argument('--critic-lr', type=float, default=1e-2, help='learning rate of critic')
    parser.add_argument('--steps-before-learn', type=int, default=5e4,
                        help='steps to be executed before agents start to learn')
    parser.add_argument('--learn-interval', type=int, default=100,
                        help='maddpg will only learn every this many steps')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save model once every time this many episodes are completed')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    args = parser.parse_args()
    return args

def get_running_reward(reward_array: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(reward_array)
        for i in range(window - 1):
            running_reward[i] = np.mean(reward_array[:i + 1])
        for i in range(window - 1, len(reward_array)):
            running_reward[i] = np.mean(reward_array[i - window + 1:i + 1])
        return running_reward

def run_experiment(args=None, world_args=None):
    # list params/ args    
    # pass default args unless specified 
    print(args)
    if args is None:
            args = default_args()
    # get default args and change any specified args
    else:
        args = get_args(args)
    print(args)
    # args = parser.parse_args()
    start = time()
    folder_name = args.env
    # create unique folder name based on world args 
    if world_args is not None and args.env == 'simple_tag_goty_edition':
        folder_name = '_'.join([f'{key}={value}' for key, value in world_args.items()])

    # create folder to save result
    env_dir = os.path.join('results', folder_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    res_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(res_dir)
    model_dir = os.path.join(res_dir, 'model')
    os.makedirs(model_dir)

    # create environment
    scenario = scenarios.load(f'{args.env}.py').Scenario()

    # if using goty edition, set world arguments from kwargs
    if args.env == 'simple_tag_goty_edition' and world_args is not None:
        world = scenario.make_world(**world_args)
    # otherwise run as normal without kwargs
    else:
        world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    # get dimension info about observation and action
    obs_dim_list = []
    for obs_space in env.observation_space:  # continuous observation
        obs_dim_list.append(obs_space.shape[0])  # Box
    act_dim_list = []
    for act_space in env.action_space:  # discrete action
        act_dim_list.append(act_space.n)  # Discrete

    maddpg = MADDPG(obs_dim_list, act_dim_list, args.buffer_capacity, args.actor_lr, args.critic_lr, res_dir)

    total_step = 0
    total_reward = np.zeros((args.episode_num, env.n))  # reward of each agent in each episode
    for episode in range(args.episode_num):
        obs = env.reset()
        # record reward of each agent in this episode
        episode_reward = np.zeros((args.episode_length, env.n))
        for step in range(args.episode_length):  # interact with the env for an episode
            actions = maddpg.select_action(obs)
            next_obs, rewards, dones, infos = env.step(actions)
            episode_reward[step] = rewards
            # env.render()
            total_step += 1

            maddpg.add(obs, actions, rewards, next_obs, dones)
            # only start to learn when there are enough experiences to sample
            if total_step > args.steps_before_learn:
                if total_step % args.learn_interval == 0:
                    maddpg.learn(args.batch_size, args.gamma)
                    maddpg.update_target(args.tau)
                if episode % args.save_interval == 0:
                    torch.save([agent.actor.state_dict() for agent in maddpg.agents],
                               os.path.join(model_dir, f'model_{episode}.pt'))

            obs = next_obs

        # episode finishes
        # calculate cumulative reward of each agent in this episode
        cumulative_reward = episode_reward.sum(axis=0)
        total_reward[episode] = cumulative_reward
        print(f'episode {episode + 1}: cumulative reward: {cumulative_reward}, '
              f'sum reward: {sum(cumulative_reward)}')

    # all episodes performed, training finishes
    # save agent parameters
    torch.save([agent.actor.state_dict() for agent in maddpg.agents], os.path.join(res_dir, 'model.pt'))
    # save training reward
    np.save(os.path.join(res_dir, 'rewards.npy'), total_reward)

    # plot result
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent in range(env.n):
        ax.plot(x, total_reward[:, agent], label=agent)
        ax.plot(x, get_running_reward(total_reward[:, agent]))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve {folder_name}'
    ax.set_title(title)
    plt.savefig(os.path.join(res_dir, title))

    print(f'training finishes, time spent: {datetime.timedelta(seconds=int(time() - start))}')

if __name__ == '__main__':
    # run_experiment(args=no_adv_sharing_args)
    # run_experiment(args=big_bounds_args)
    main_args = {'env': 'simple_tag_goty_edition'}
    world_args = {
    'out_of_bound_punishment': 100,
    'hard_boundary': True,
    'good_collaborative': True,
    'bad_collaborative': False
    }
    run_experiment(args=main_args, world_args=world_args)