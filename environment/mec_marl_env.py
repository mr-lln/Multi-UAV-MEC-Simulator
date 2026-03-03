
import gym
from gym import spaces
import numpy as np
from .space_def import circle_space
from .space_def import onehot_space
from .space_def import sum_space
import logging

logging.basicConfig(level=logging.WARNING)

def get_circle_plot(pos, r):
    x_c = np.arange(-r, r, 0.01)
    up_y = np.sqrt(r**2 - np.square(x_c))
    down_y = - up_y
    x = x_c + pos[0]
    y1 = up_y + pos[1]
    y2 = down_y + pos[1]
    return [x, y1, y2]


class MEC_MARL_ENV(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, world, alpha=0.5, beta=0.2, aggregate_reward=False, discrete=True,
                 reset_callback=None, info_callback=None, done_callback=None,energy_weight=0.3):
        self.world = world
        self.obs_r = world.obs_r
        self.move_r = world.move_r
        self.collect_r = world.collect_r
        self.max_buffer_size = self.world.max_buffer_size
        self.agents = self.world.agents
        self.agent_num = self.world.agent_count
        self.sensor_num = self.world.sensor_count
        self.sensors = self.world.sensors
        self.DS_map = self.world.DS_map
        self.map_size = self.world.map_size
        self.DS_state = self.world.DS_state
        self.alpha = alpha
        self.beta = beta

        self.reset_callback = reset_callback
        self.info_callback = info_callback
        self.done_callback = done_callback


        self.aggregate_reward = aggregate_reward
        self.discrete_flag = discrete
        self.state = None
        self.time = 0
        self.images = []

        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            if self.discrete_flag:
                act_space = spaces.Tuple((circle_space.Discrete_Circle(
                    agent.move_r), onehot_space.OneHot(self.max_buffer_size), sum_space.SumOne(self.agent_num), onehot_space.OneHot(self.max_buffer_size)))
                obs_space = spaces.Tuple((spaces.MultiDiscrete(
                    [self.map_size, self.map_size]), spaces.Box(0, np.inf, [agent.obs_r * 2, agent.obs_r * 2, 2])))
                self.action_space.append(act_space)
                self.observation_space.append(obs_space)
        self.render()
        self.energy_weight = energy_weight
        self.time_step = 0

    def step(self, agent_action, center_action):
        obs = []
        reward = []
        done = []
        age = []
        info = {'n': [], 'energy': {}}
        self.agents = self.world.agents

        for i, agent in enumerate(self.agents):
            self._set_action(agent_action[i], center_action, agent)

        self.world.step()

        for agent in self.agents:
            agent.calculate_energy_from_actions()

        for agent in self.agents:
            obs.append(self.get_obs(agent))
            done.append(self._get_done(agent))
            reward.append(self._get_cost())
            age.append(self._get_age())
            info['n'].append(self._get_info(agent))
        self.state = obs
        self.time_step += 1

        reward_sum = np.sum(reward)
        if self.aggregate_reward:
            reward = [reward_sum] * self.agent_num
        return self.state, reward, age,done, info

    def reset(self):
        self.world.finished_data = []
        for sensor in self.sensors:
            sensor.data_buffer = []
            sensor.collect_state = False
        for agent in self.agents:
            agent.idle = True
            agent.data_buffer = {}
            agent.total_data = {}
            agent.done_data = []
            agent.collecting_sensors = []
        for agent in self.agents:
            agent.reset_energy_stats()

        self.time_step = 0



    def _set_action(self, act, center_action, agent):
        agent.action.move = np.zeros(2)
        agent.action.execution = act[1]
        agent.action.bandwidth = center_action[agent.no]
        if agent.movable and agent.idle:
            if np.linalg.norm(act[0]) > agent.move_r:
                act[0] = [int(act[0][0] * agent.move_r / np.linalg.norm(act[0])), int(act[0][1] * agent.move_r / np.linalg.norm(act[0]))]
            if not np.count_nonzero(act[0]) and np.random.rand() > 0.5:
                mod_x = np.random.normal(loc=0, scale=1)
                mod_y = np.random.normal(loc=0, scale=1)
                mod_x = int(min(max(-1, mod_x), 1) * agent.move_r / 2)
                mod_y = int(min(max(-1, mod_y), 1) * agent.move_r / 2)
                act[0] = [mod_x, mod_y]
            agent.action.move = np.array(act[0])
            new_x = agent.position[0] + agent.action.move[0]
            new_y = agent.position[1] + agent.action.move[1]
            if new_x < 0 or new_x > self.map_size - 1:
                agent.action.move[0] = -agent.action.move[0]
            if new_y < 0 or new_y > self.map_size - 1:
                agent.action.move[1] = -agent.action.move[1]
            agent.position += agent.action.move

        if agent.offloading_idle:
            agent.action.offloading = act[2]

    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    def get_obs(self, agent):
        obs = np.zeros([agent.obs_r * 2 + 1, agent.obs_r * 2 + 1, 2])
        lu = [max(0, agent.position[0] - agent.obs_r),
              min(self.map_size, agent.position[1] + agent.obs_r + 1)]
        rd = [min(self.map_size, agent.position[0] + agent.obs_r + 1),
              max(0, agent.position[1] - agent.obs_r)]
        ob_lu = [agent.obs_r - agent.position[0] + lu[0],
                 agent.obs_r - agent.position[1] + lu[1]]
        ob_rd = [agent.obs_r + rd[0] - agent.position[0],
                 agent.obs_r + rd[1] - agent.position[1]]
        for i in range(ob_rd[1], ob_lu[1]):
            map_i = rd[1] + i - ob_rd[1]
            obs[i][ob_lu[0]:ob_rd[0]] = self.DS_state[map_i][lu[0]:rd[0]]
        agent.obs = obs
        return obs

    def get_statemap(self):
        sensor_map = np.ones([self.map_size, self.map_size, 2])
        agent_map = np.ones([self.map_size, self.map_size, 2])
        for sensor in self.sensors:
            sensor_map[int(sensor.position[1])][int(sensor.position[0])][0] = sum([i[0] for i in sensor.data_buffer])
            sensor_map[int(sensor.position[1])][int(sensor.position[0])][1] = sum([i[1] for i in sensor.data_buffer]) / max(len(sensor.data_buffer), 1)
        for agent in self.agents:
            agent_map[int(agent.position[1])][int(agent.position[0])][0] = sum([i[0] for i in agent.done_data])
            agent_map[int(agent.position[1])][int(agent.position[0])][1] = sum([i[1] for i in agent.done_data]) / max(len(agent.done_data), 1)
        return sensor_map, agent_map

    def get_center_state(self):
        buffer_list = np.zeros([self.agent_num, 2, self.max_buffer_size])
        pos_list = np.zeros([self.agent_num, 2])
        for i, agent in enumerate(self.agents):
            pos_list[i] = agent.position
            for j, d in enumerate(agent.done_data):
                buffer_list[i][0][j] = d[0]
                buffer_list[i][1][j] = d[1]
        return buffer_list, pos_list

    def get_buffer_state(self):
        exe = []
        done = []
        for agent in self.agents:
            exe.append(len(agent.total_data))
            done.append(len(agent.done_data))
        return exe, done

    def _get_done(self, agent):
        if self.done_callback is None:
            return 0
        return self.done_callback(agent, self.world)

    def _get_age(self):
        return np.mean(list(self.world.sensor_age.values()))

    def _get_cost(self):
        return np.mean(list(self.world.sensor_age.values()))

    def close(self):
        return None
