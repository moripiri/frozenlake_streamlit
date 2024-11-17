import streamlit as st
import pandas as pd
#https://github.com/streamlit/streamlit/issues/1291
st.set_page_config(page_title='RL on FrozenLake',
                   layout='wide',
                   initial_sidebar_state='expanded')
st.sidebar.success("Select a demo above.")

st.title('Reinforcement Learning on FronzenLake 🧊')
st.divider()

