import os

from gym import utils, spaces
from mjrl.envs import mujoco_env
import mujoco_py
import numpy as np


ADD_BONUS_REWARDS = True
DEFAULT_SIZE = 128


class RelocateEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, use_full_state=False):
        self.target_obj_sid = 0
        self.S_grasp_sid = 0
        self.obj_bid = 0
        self.target_pos = np.zeros(3)

        if use_full_state:
            self._get_obs = self.get_env_full_state
        else:
            # self._get_obs = self.get_obs
            self._get_obs = self.get_env_state

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir + '/assets/DAPG_relocate.xml', 5)

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[
        self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0') + 1, :3] = np.array(
            [10, 0, 0])
        self.sim.model.actuator_gainprm[
        self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0') + 1, :3] = np.array(
            [1, 0, 0])
        self.sim.model.actuator_biasprm[
        self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0') + 1, :3] = np.array(
            [0, -10, 0])
        self.sim.model.actuator_biasprm[
        self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0') + 1, :3] = np.array(
            [0, -1, 0])

        self.target_obj_sid = self.sim.model.site_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:, 1])
        self.action_space.low = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:, 0])

        obs = self.get_obs()
        state = self.get_env_state()
        full_state = self.get_env_full_state()
        self.obs_dim = obs.size
        self.state_dim = state.size
        self.full_state_dim = full_state.size

        obs_high = np.inf * np.ones(self.obs_dim)
        obs_low = -obs_high
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        state_high = np.inf * np.ones(self.state_dim)
        state_low = -state_high
        self.state_space = spaces.Box(state_low, state_high, dtype=np.float32)
        full_state_high = np.inf * np.ones(self.full_state_dim)
        full_state_low = -full_state_high
        self.full_state_space = spaces.Box(full_state_low, full_state_high, dtype=np.float32)

        self.viewer = None
        self._viewers = {}

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.target_pos.ravel()

        reward = -0.1 * np.linalg.norm(palm_pos - obj_pos)  # take hand to object
        if obj_pos[2] > 0.04:  # if object off the table
            reward += 1.0  # bonus for lifting the object
            reward += -0.5 * np.linalg.norm(palm_pos - target_pos)  # make hand go to target
            reward += -0.5 * np.linalg.norm(obj_pos - target_pos)  # make object go to target

        if ADD_BONUS_REWARDS:
            if np.linalg.norm(obj_pos - target_pos) < 0.1:
                reward += 10.0  # bonus for object close to target
            if np.linalg.norm(obj_pos - target_pos) < 0.05:
                reward += 20.0  # bonus for object "very" close to target

        goal_achieved = True if np.linalg.norm(obj_pos - target_pos) < 0.1 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return np.concatenate([qp[:-6], palm_pos - obj_pos, palm_pos - target_pos, obj_pos - target_pos])

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid, 0] = self.np_random.uniform(low=-0.15, high=0.15)
        self.model.body_pos[self.obj_bid, 1] = self.np_random.uniform(low=-0.15, high=0.3)
        self.target_pos[0] = self.np_random.uniform(low=-0.2, high=0.2)
        self.target_pos[1] = self.np_random.uniform(low=-0.2, high=0.2)
        self.target_pos[2] = self.np_random.uniform(low=0.15, high=0.35)
        self.sim.forward()
        return self._get_obs()

    def get_env_full_state(self):
        return self.get_env_state()

    def full_state_to_state(self, full_states):
        assert full_states.ndim == 2
        return full_states

    def full_state_to_obs(self, full_states):
        assert full_states.ndim == 2
        qp = full_states[:, :36]
        obj_pos = full_states[:, 102:105]
        palm_pos = full_states[:, 105:108]
        target_pos = full_states[:, 108:111]
        return np.concatenate([qp[:, :-6], palm_pos - obj_pos, palm_pos - target_pos, obj_pos - target_pos], axis=-1)

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.target_pos.ravel()
        return np.concatenate([qp, qv, hand_qpos, obj_pos, palm_pos, target_pos])

    def set_env_state(self, state):
        qp = state[:36]
        qv = state[36:72]
        self.target_pos = state[108:111]
        self.set_state(qp, qv)

    def set_env_full_state(self, full_state):
        self.set_env_state(full_state)

    def mj_viewer_setup(self):
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if object close to target for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage

    def render(self,
               mode='human',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'fixed'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == 'rgb_array':
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.mj_viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}
