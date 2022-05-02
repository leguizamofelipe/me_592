import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
# from mpl_toolkits import mplot3d
import time
import os

from matplotlib.patches import Rectangle

class EpisodeHistory():
    def __init__(self, ep_no, init, target) -> None:
        self.ep_no = ep_no
        self.endpoint_history = []
        self.reward_history = []
        self.cumulative_reward_history = []
        self.first_timestep = True
        self.target = target
        self.init = init
        self.start_time = time.time()
        self.alpha_list = []
        self.beta_list = []
        self.gamma_list = []
        self.alpha_list_norm = []
        self.beta_list_norm = []
        self.gamma_list_norm = []
    def record_position(self, endpoint):
        self.endpoint_history.append(endpoint)
    def record_reward(self, reward):
        self.reward_history.append(reward)
        if self.first_timestep:
            self.prev_reward = 0
            self.first_timestep = False
        else:
            self.prev_reward = self.cumulative_reward_history[-1]
        self.cumulative_reward_history.append(reward + self.prev_reward)
    def total_reward(self):
        return sum(self.reward_history)
    def record_steps(self, steps):
        self.final_steps = steps
    def save_episode(self, save_dir, boxes = None, addendum = ''):
        # ax = plt.axes(projection = '3d')
        ax= plt.axes()
        ax.set_xlim([-1.5,1.5]) # Was 0.5
        ax.set_ylim([-1.5,1.5]) # Was 0.5
        # ax.set_zlim([0,1])
        x_list = [point.x for point in self.endpoint_history]
        y_list = [point.y for point in self.endpoint_history]
        
        if boxes is None:
            pass
        else:
            for box in boxes:
                rect = Rectangle((box.x, box.y), 0.2, 0.2)
                # ax.add_patch(rect)

        # z_list = [point.z for point in self.endpoint_history]
        # ax.plot3D(x_list, y_list, z_list, 'blue')
        # ax.scatter3D(self.init.x, self.init.y, self.init.z, color = 'r')
        # ax.scatter3D(self.target.x, self.target.y, self.target.z, color = 'g')
        ax.plot(x_list, y_list, 'blue')
        ax.scatter(self.init.x, self.init.y, color = 'r')
        ax.scatter(self.target.x, self.target.y, color = 'g')
        ax.set_title(f'n_steps: {len(self.endpoint_history)} | reward: {self.cumulative_reward_history[-1]}' + addendum)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{int(time.time())}.png'))
        plt.close()

        plt.subplot(2, 1, 1)
        plt.plot(self.alpha_list, color='b')
        plt.title('Alpha')

        plt.subplot(2, 1, 2)
        plt.plot(self.alpha_list_norm, color='b')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'alpha.png'))
        plt.close()

        plt.subplot(2, 1, 1)
        plt.plot(self.beta_list, color='r')
        plt.title('Beta')

        plt.subplot(2, 1, 2)
        plt.plot(self.beta_list_norm, color='r')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'beta.png'))
        plt.close()

        plt.subplot(2, 1, 1)
        plt.plot(self.gamma_list, color='g')
        plt.title('Gamma')

        plt.subplot(2, 1, 2)
        plt.plot(self.gamma_list_norm, color='g')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'gamma.png'))
        plt.close()

        '''
        ax = plt.axes()
        ax.plot(self.alpha_list, label='Alpha', color='b')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'alpha.png'))
        plt.title('Alpha')
        plt.close()

        ax = plt.axes()
        ax.plot(self.beta_list, label='Beta', color='r')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'beta.png'))
        plt.title('Beta')
        plt.close()

        ax = plt.axes()
        ax.plot(self.gamma_list, label='Gamma', color='g')
        plt.title('Gamma')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'gamma.png'))
        plt.close()
        '''
        
    def record_proportions(self, alpha, beta, gamma, reward):
        self.alpha_list.append(alpha)
        self.beta_list.append(beta)
        self.gamma_list.append(gamma)
        self.alpha_list_norm.append(alpha/reward)
        self.beta_list_norm.append(beta/reward)
        self.gamma_list_norm.append(gamma/reward)

        