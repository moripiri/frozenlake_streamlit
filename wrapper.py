import gymnasium as gym
import matplotlib.pyplot as plt
import math
from visualize import visualize_policy, visualize_q, visualize_model, visualize_v
from utils import figure_to_array

class JupyterRender(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.env_width = int(math.sqrt(self.env.observation_space.n))

    def render(self, title='', v=None, q=None, policy=None, model_r=None, model_ns=None):
        viz_list = {}
        if v is not None:
            viz_list['v'] = v
        if q is not None:
            viz_list['q'] = q
        if policy is not None:
            viz_list['policy'] = policy
        if model_r is not None:
            viz_list['model_r'] = model_r
        if model_ns is not None:
            viz_list['model_ns'] = model_ns
        plt.close('all')

        fig = plt.figure(figsize=(self.env_width*2, self.env_width*2))

        ax_list = [fig.add_subplot(2, 2, 1)]
        img = ax_list[0].imshow(
            self.env.render())  # prepare to render the environment by using matplotlib and ipython display
        ax_list[0].set_title(title)

        pos = 2
        for i in range(pos, 2 + len(viz_list)):
            ax_list.append(fig.add_subplot(2, 2, i))

        ax_index = 1
        for key, value in viz_list.items():
            if key == 'policy':
                visualize_policy(value, ax_list[ax_index], self.env.nrow, self.env.ncol)
            elif key == 'v':
                visualize_v(value, ax_list[ax_index], self.env.nrow, self.env.ncol)
            elif key == 'q':
                visualize_q(value, ax_list[ax_index], self.env.nrow, self.env.ncol)
            else:
                if key == 'model_r':
                    title = 'Reward Model'
                elif key == 'model_ns':
                    title = 'Next State Model'
                visualize_model(value, ax_list[ax_index], self.env.nrow, self.env.ncol, title)

            ax_index += 1

        for ax in ax_list:
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

        return fig