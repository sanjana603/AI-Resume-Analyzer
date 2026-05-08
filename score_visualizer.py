import matplotlib.pyplot as plt
import streamlit as st


def plot_chart(score):

    fig, ax = plt.subplots(figsize=(6, 1))

    colors = ['#ff4b4b', '#ffa726', '#0f9d58']

    color_index = min(int(score // 33), 2)

    ax.barh([0], [score], color=colors[color_index])

    ax.set_xlim(0, 100)

    ax.set_xlabel("Match Percentage")

    ax.set_yticks([])

    ax.set_title("Resume Job Match")

    st.pyplot(fig)