import gymnasium as gym
import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
from wrapper import JupyterRender
st.set_page_config(page_title="sarsa(λ)", page_icon="🎲")

st.title("5. SARSA(λ)")

st.markdown(
    '''
    This demo shows SARSA(λ) in FrozenLake-v1. 

    Enjoy!
    '''
)
map_size = st.sidebar.radio(
        "map_size",
        index=0,
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
lamda = st.sidebar.number_input(
        "lamda",
        value=0.5,
        min_value=0.,
        max_value=1.,
        step=0.01,
        key="lamda",
        help="eligibility discount"
    )
alpha = st.sidebar.number_input(
        "alpha",
        value=0.1,
        min_value=0.,
        max_value=1.,
        step=0.01,
        key="alpha",
        help="learning rate"
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
speed_display = {0.5: 'slow', 0.1: 'normal', 0.01: 'fast'}
speed = st.sidebar.pills(label="Speed", options=[0.5, 0.1, 0.01], format_func=lambda x: speed_display[x],  default=0.1)


episode = 0
reset = st.button("Play")
if reset:
    env = JupyterRender(gym.make("FrozenLake-v1", map_name=map_size, is_slippery=slippery, render_mode='rgb_array'))
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    q = np.zeros([state_dim, action_dim])
    rewards, successes = [], []
    with st.empty():
        for episode in range(max_episode):
            eligibility = np.zeros([state_dim, action_dim])

            s, _ = env.reset()
            if np.random.random() < eps:
                a = np.random.randint(low=0, high=action_dim - 1)
            else:
                a = np.argmax(q[s, :])

            dones = False
            success = False
            timestep = 0
            total_reward = 0
            while not dones:
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

                if np.random.random() < eps:
                    na = np.random.randint(low=0, high=action_dim - 1)
                else:
                    na = np.argmax(q[ns, :])

                # sarsa
                delta = r + gamma * q[ns, na] - q[s, a]
                eligibility[s, a] += 1

                # eligibility update
                for i in range(state_dim):
                    for j in range(action_dim):
                        q[i, j] += alpha * delta * eligibility[i, j]
                        eligibility[i, j] = gamma * lamda * eligibility[i, j]

                total_reward += r

                with st.container():
                    st.subheader(f"Episode {episode}, Step {timestep}")
                    st.pyplot(env.render(q=q), clear_figure=True)

                s = ns
                a = na
                dones = d or t
                timestep += 1

                if timestep == 1000:
                    dones = True

                time.sleep(speed)

            rewards.append(total_reward)
            successes.append(success)

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

