import gymnasium as gym
import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
from wrapper import JupyterRender
st.set_page_config(page_title="monte-carlo control", page_icon="🎲")

st.title("4. Monte-carlo control")

st.markdown(
    '''
    This demo shows Monte-carlo control in FrozenLake-v1. 

    Enjoy!
    '''
)
map_size = st.sidebar.radio(
        "map_size",
        key="map_size",
        options=["4x4", "8x8"],
    )
slippery = st.sidebar.checkbox("slippery", key='slippery ice', help='if ice is slippery, player will move randomly 25%')

max_episode = st.sidebar.number_input(
        "max_episode",
        value=10,
        min_value=1,
        max_value=1000,
        step=1,
        key="max_episode",
        help="maximum episodes to run the policy"
    )

gamma = st.sidebar.number_input(
        "gamma",
        value=0.9,
        min_value=0.,
        max_value=1.,
        step=0.01,
        key="gamma",
        help="future reward discount"
    )

eps = st.sidebar.number_input(
        "eps",
        value=0.1,
        min_value=0.,
        max_value=1.,
        step=0.01,
        key="eps",
        help="random action probability"
    )
reach_hole_reward = st.sidebar.number_input(
        "reach_hole_reward",
        value=-1.,
        min_value=-100.,
        max_value=100.,
        step=0.01,
        key="reach_hole_reward",
        help="reward to give when player reaches the hole"
    )
heading_wall_reward = st.sidebar.number_input(
        "heading_wall_reward",
        value=-1.,
        min_value=-100.,
        max_value=100.,
        step=0.01,
        key="heading_wall_reward",
        help="reward to give when player is heading the wall"
    )
staying_reward = st.sidebar.number_input(
        "staying_reward",
        value=-0.01,
        min_value=-100.,
        max_value=100.,
        step=0.01,
        key="staying_reward",
        help="reward to give when player is on the ice"
    )
speed_display = {0.1: 'slow', 0.05: 'normal', 0.005: 'fast'}
speed = st.sidebar.pills(label="Speed", options=[0.1, 0.05, 0.005], format_func=lambda x: speed_display[x],  default=0.05)

episode = 0
reset = st.button("Play")
if reset:
    env = JupyterRender(gym.make("FrozenLake-v1", map_name=map_size, is_slippery=slippery, render_mode='rgb_array'))
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    pi = np.ones([state_dim, action_dim]) / action_dim
    q = np.zeros([state_dim, action_dim])
    rewards, successes = [], []

    with st.empty():
        for episode in range(max_episode):
            s, _ = env.reset()

            dones = False
            success = False
            timestep = 0
            total_reward = 0

            trajectory = []
            returns = [[[] for _ in range(action_dim)] for _ in range(state_dim)]

            while not dones:
                a = np.random.choice(list(range(action_dim)), p=pi[s])

                ns, r, d, t, info = env.step(a)

                if r == 0:
                    # prevent staying in the ground forever
                    r += staying_reward

                if s == ns:
                    # prevent heading into the war
                    r += heading_wall_reward

                if d and ns != state_dim - 1:
                    # Failure
                    r += reach_hole_reward

                if d and ns == state_dim - 1:
                    success = True

                # q-learning
                trajectory.append({'s': s, 'a': a, 'r': r})
                total_reward += r

                with st.container():
                    st.subheader(f"Episode {episode}, Step {timestep}")
                    st.pyplot(env.render(q=q, policy=pi), clear_figure=True)

                s = ns
                dones = d or t
                timestep += 1

                if timestep == 1000:
                    dones = True

                time.sleep(speed)

            rewards.append(total_reward)
            successes.append(success)

            trajectory.reverse()

            G = 0

            traj_sa_list = list(map(lambda x: [x['s'], x['a']], trajectory))

            for traj in range(len(trajectory)):
                G = gamma * G + trajectory[traj]['r']
                cur_s = trajectory[traj]['s']
                cur_a = trajectory[traj]['a']

                if [cur_s, cur_a] not in traj_sa_list[traj + 1:]:
                    returns[cur_s][cur_a].append(G)

                    q[cur_s][cur_a] = sum(returns[cur_s][cur_a]) / len(returns[cur_s][cur_a])

                    optimal_action = np.argmax(q, axis=1)

                    for i in range(state_dim):
                        for j in range(action_dim):
                            if optimal_action[i] == j:
                                pi[i][j] = 1 - eps + eps / action_dim
                            else:
                                pi[i][j] = eps / action_dim

    fig = plt.figure(figsize=(10, 5))
    plt.plot(rewards, markeredgecolor='red', markerfacecolor='red', marker='.', markersize=20)
    plt.plot([m for m, success in enumerate(successes) if success], [reward for n, reward in enumerate(rewards) if successes[n]],
             lw=0, markeredgecolor='green', markerfacecolor='green', marker='.', markersize=20, label='Success')

    plt.xticks(list(range(max_episode)))
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend(loc='upper left')
    plt.title("Episode Rewards")
    plt.grid()
    st.pyplot(fig, clear_figure=True)

