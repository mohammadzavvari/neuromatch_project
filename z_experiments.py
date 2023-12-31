import argparse
import os
import time


import numpy as np
import torch
from matplotlib import pyplot as plt
from multiagent import scenarios
from multiagent.environment import MultiAgentEnv

# for making a gif
import imageio
import glob
import pyglet
from MADDPG import MADDPG
import re

# Simple experiments, vary the arguments of simple_tag and see what happens.


def default_args():
    "returns default args for simple_tag"
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="simple_tag_no_adv_sharing", help='name of the environment',
                        choices=['simple_adversary', 'simple_crypto', 'simple_push', 'simple_reference',
                                 'simple_speaker_listener', 'simple_spread', 'simple_tag',
                                 'simple_world_comm', 'simple_tag_colab'])
    parser.add_argument('--folder', type=str, default='1', help='name of the folder where model is saved')
    parser.add_argument('--episode-length', type=int, default=50, help='steps per episode')
    parser.add_argument('--episode-num', type=int, default=2, help='total number of episode')
    args = parser.parse_args()
    return args

DEFAULT_ARGS = {'env': 'simple_tag_no_adv_sharing', 'folder': '1', 'episode_length': 50, 'episode_num': 2}

def arg_parser(args_dict):
    "convert arg dict to argparse object"
    parser = argparse.ArgumentParser()
    for key, value in args_dict.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    return parser.parse_args()

def get_args(new_args, default_args=DEFAULT_ARGS):
    "return default args with any specified arguments updates as argparse object"
    args_dict = default_args
    for key, value in new_args.items():
        args_dict[key] = value
    return arg_parser(args_dict)

def run_experiment():
    # list params/ args    
    pass

# def create_gif(input_folder, output_gif):
#     # Gather all frames, in order
#     frame_files = sorted(glob.glob(os.path.join(input_folder, '*.png')))
    
#     # Read frames into a list
#     frames = [imageio.imread(frame_file) for frame_file in frame_files]
    
#     # Save frames as a GIF
#     imageio.mimsave(output_gif, frames)

def sort_key(filename):
    match = re.search(r'episode_(\d+)_step_(\d+)', filename)
    episode, step = map(int, match.groups())
    return (episode, step)  # Sort by episode first, then step within each episode

def create_gif(input_folder, output_gif):
    # Gather all frames, in order
    frame_files = glob.glob(os.path.join(input_folder, 'episode_*.png'))
    
    # Custom sorting function
    # Sort the filenames using the custom sorting function
    frame_files = sorted(frame_files, key=sort_key)

    # Read frames into a list
    frames = [imageio.imread(frame_file) for frame_file in frame_files]
    
    # Save frames as a GIF
    imageio.mimsave(output_gif, frames)
    
def evaluate_model(args=None, world_args=None, save_video=True, folder_name=None):
    if args is None:
        args = default_args()

    # scenario = scenarios.load(f'{args.env}.py').Scenario()
    # world = scenario.make_world()
    # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    # create env
    scenario = scenarios.load(f'{args.env}.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    # get dimension info about observation and action
    obs_dim_list = []
    for obs_space in env.observation_space:  # continuous observation
        obs_dim_list.append(obs_space.shape[0])  # Box
    act_dim_list = []
    for act_space in env.action_space:  # discrete action
        act_dim_list.append(act_space.n)  # Discrete

    maddpg = MADDPG(obs_dim_list, act_dim_list, 0, 0, 0)
    if folder_name is None:
        folder_name = args.env
        # create unique folder name based on world args 
        # we need to specify the folder number manually still
        if world_args is not None and args.env == 'simple_tag_goty_edition':
            folder_name = '_'.join([f'{key}={value}' for key, value in world_args.items()])

    model_dir = os.path.join('results', folder_name, args.folder)
    # make model_dir if model_dir does not exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    assert os.path.exists(model_dir)

    data = torch.load(os.path.join(model_dir, 'model.pt'))
    for agent, actor_parameter in zip(maddpg.agents, data):
        agent.actor.load_state_dict(actor_parameter)
    print(f'MADDPG load model.pt from {model_dir}')

    frame_dir = os.path.join(model_dir, "frames")

    # Create directories if they don't exist
    os.makedirs(frame_dir, exist_ok=True)

    total_reward = np.zeros((args.episode_num, env.n))  # reward of each episode
    frame_no = 0
    for episode in range(args.episode_num):
        obs = env.reset()
        # record reward of each agent in this episode
        episode_reward = np.zeros((args.episode_length, env.n))
        for step in range(args.episode_length):  # interact with the env for an episode
            actions = maddpg.select_action(obs)
            next_obs, rewards, dones, infos = env.step(actions)
            episode_reward[step] = rewards
            if save_video:
                # render output for rgb array is a list of frames
                # to view, we just need the first frame
                frame = env.render(mode='rgb_array')[0]
                name = f'episode_{episode}_step_{step}.png'
                plt.imsave(os.path.join(model_dir, "frames", name), frame)
                plt.close("all")
            else:
                env.render()
            time.sleep(0.02)
            obs = next_obs
        pyglet.app.exit()
        # episode finishes
        # calculate cumulative reward of each agent in this episode
        cumulative_reward = episode_reward.sum(axis=0)
        total_reward[episode] = cumulative_reward
        print(f'episode {episode + 1}: cumulative reward: {cumulative_reward}')

    # Create gif
    create_gif(os.path.join(model_dir, "frames"), os.path.join(model_dir, "animation.gif"))
    
    os.makedirs("animations", exist_ok=True)
    create_gif(os.path.join(model_dir, "frames"), os.path.join("animations", folder_name+".gif"))

    
    # all episodes performed, evaluate finishes
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent in range(env.n):
        ax.plot(x, total_reward[:, agent], label=agent)
        # ax.plot(x, get_running_reward(total_reward[:, agent]))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'evaluating result of maddpg solve {args.env}'
    ax.set_title(title)
    plt.savefig(os.path.join(model_dir, title))

    # close the environment!
    env.close()
    plt.close('all')


def evaluate_trained_models(episode_length=100, episode_num=5):
    "evaluate all trained models and output gifs"
    args_dict = {'episode_length': episode_length, 'episode_num': episode_num}
    env_folder_dict = [
        {'env': 'simple_tag', 'folder': '1'},
        {'env': 'simple_tag_4g_1b', 'folder': '2'},
        {'env': 'simple_tag_4g_1b_colab', 'folder': '1'},
        {'env': 'simple_tag_4g2b_goodcolab', 'folder': '3'},
        {'env': 'simple_tag_big_bounds', 'folder': '1'},
        {'env': 'simple_tag_colab', 'folder': '2'},
        {'env': 'simple_tag_no_adv_sharing', 'folder': '4'},
    ]
    for env_folder in env_folder_dict:
        main_args_dict = {**env_folder, **args_dict}
        print(main_args_dict)
        main_args = get_args(main_args_dict)
        evaluate_model(main_args, save_video=True)


        
def evaluate_new_models(episode_length=100, episode_num=5, folder_names=None):
    "New models with goty edition"
    args_dict = {'episode_length': episode_length, 'episode_num': episode_num}
    env_folder_dict = {'env': 'simple_tag_goty_edition', 'folder': '1'}
    for folder_name in folder_names:
        main_args_dict = {**env_folder_dict, **args_dict}
        print(main_args_dict)
        main_args = get_args(main_args_dict)
        evaluate_model(main_args, save_video=True, folder_name=folder_name)


    

if __name__ == "__main__":
    args = default_args()
    # Specify folder using arguments
    # main_args = {'env': 'simple_tag_no_adv_sharing', 'folder': '4', 'episode_length': 100, 'episode_num': 3}
    # main_args = get_args(main_args)
    # evaluate_model(main_args, save_video=True)

    # main_args = {'env': 'simple_tag_big_bounds', 'folder': '1', 'episode_length': 100, 'episode_num': 3}
    # main_args = get_args(main_args)
    # evaluate_model(main_args, save_video=True)


    # main_args = {'env': 'simple_tag_goty_edition', 'folder': '1', 'episode_length': 100, 'episode_num': 3}
    # main_args = get_args(main_args)
    # world_args = {'out_of_bound_punishment':100,
    #               'hard_boundary':True,
    #               'good_collaborative': True,
    #               'bad_collaborative' : False}
    # evaluate_model(main_args, world_args, save_video=True)
    # evaluate_trained_models(episode_length=100, episode_num=5)

    dan_folders = ["no_colab", 
                  "no_colab_soft_bounds", 
                  "good_sharing",
                  "good_sharing_soft_bounds"]
    jasvin_folders = ["bad_sharing", "bad_sharing_soft_bounds"] # ashutosh
    mrugsen_folders = ["good_and_bad_sharing", "good_and_bad_sharing_soft_bounds"]
    mohammad_folders = ["all_share_together", "all_share_together_soft_bounds"] # dan
    shape_folders = ["shape_reward", "no_colab_shape_reward"] # dan macbook

    evaluate_new_models(episode_length=100, episode_num=3, 
                        folder_names=[mohammad_folders[1]])