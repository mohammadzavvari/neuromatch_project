from pettingzoo.mpe import simple_tag_v3

env = simple_tag_v3.env(render_mode="human", max_cycles=5_000)

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(
            agent        ).sample()  # this is where you would insert your policy

    env.step(action)
env.close()
