import gymnasium as gym
import streamlit as st
import numpy as np
import time

st.set_page_config(page_title="Play", page_icon="📈")

st.title("Plotting Demo")
st.sidebar.header("How to play")
st.sidebar.write("How to play")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

env = gym.make("FrozenLake-v1", is_slippery=False)

last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    chart.add_rows(new_rows)
    last_rows = new_rows
    time.sleep(0.05)

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")