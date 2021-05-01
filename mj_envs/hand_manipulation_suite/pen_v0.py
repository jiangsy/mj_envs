import os

from gym import utils, spaces
from mj_envs.utils.quatmath import euler2quat
from mjrl.envs import mujoco_env
import mujoco_py
import numpy as np

ADD_BONUS_REWARDS = True
DEFAULT_SIZE = 128


class PenEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, use_full_state=False):
        self.target_obj_bid = 0
        self.S_grasp_sid = 0
        self.eps_ball_sid = 0
        self.obj_bid = 0
        self.obj_t_sid = 0
        self.obj_b_sid = 0
        self.tar_t_sid = 0
        self.tar_b_sid = 0
        self.pen_length = 1.0
        self.tar_length = 1.0

        if use_full_state:
            self._get_obs = self.get_env_full_state
        else:
            self._get_obs = self.get_obs
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir + '/assets/DAPG_pen.xml', 5)

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

        utils.EzPickle.__init__(self)
        self.target_obj_bid = self.sim.model.body_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.eps_ball_sid = self.sim.model.site_name2id('eps_ball')
        self.obj_t_sid = self.sim.model.site_name2id('object_top')
        self.obj_b_sid = self.sim.model.site_name2id('object_bottom')
        self.tar_t_sid = self.sim.model.site_name2id('target_top')
        self.tar_b_sid = self.sim.model.site_name2id('target_bottom')

        self.pen_length = np.linalg.norm(self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])
        self.tar_length = np.linalg.norm(self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])

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
            starting_up = False
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            starting_up = True
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)

        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        desired_loc = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid]) / self.pen_length
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid]) / self.tar_length

        # pos cost
        dist = np.linalg.norm(obj_pos - desired_loc)
        reward = -dist
        # orien cost
        orien_similarity = np.dot(obj_orien, desired_orien)
        reward += orien_similarity

        if ADD_BONUS_REWARDS:
            # bonus for being close to desired orientation
            if dist < 0.075 and orien_similarity > 0.9:
                reward += 10
            if dist < 0.075 and orien_similarity > 0.95:
                reward += 50

        # penalty for dropping the pen
        done = False
        if obj_pos[2] < 0.075:
            reward -= 5
            done = True if not starting_up else False

        goal_achieved = True if (dist < 0.075 and orien_similarity > 0.95) else False

        return self._get_obs(), reward, done, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        qp = self.data.qpos.ravel()
        obj_vel = self.data.qvel[-6:].ravel()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        desired_pos = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid]) / self.pen_length
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid]) / self.tar_length
        return np.concatenate([qp[:-6], obj_pos, obj_vel, obj_orien, desired_orien,
                               obj_pos - desired_pos, obj_orien - desired_orien])

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-1, high=1)
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.sim.forward()
        return self._get_obs()

    def get_env_full_state(self):
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        desired_pos = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid]) / self.pen_length
        desired_orien4 = self.model.body_quat[self.target_obj_bid].ravel().copy()
        desired_orien3 = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length
        return np.concatenate([qp, qv, desired_orien4, obj_pos, desired_pos, obj_orien, desired_orien3])

    def full_state_to_state(self, full_states):
        assert full_states.ndim == 2
        return full_states[:, :64]

    def full_state_to_obs(self, full_states):
        assert full_states.ndim == 2
        qp = full_states[:, :30]
        obj_vel = full_states[:, 54:60]
        obj_pos = full_states[:, 64:67]
        desired_pos = full_states[:, 67:70]
        obj_orien = full_states[:, 70:73]
        desired_orien3 = full_states[:, 73:76]
        return np.concatenate([qp[:, :-6], obj_pos, obj_vel, obj_orien, desired_orien3,
                               obj_pos - desired_pos, obj_orien - desired_orien3], axis=-1)

    def get_env_state(self):
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        desired_orien = self.model.body_quat[self.target_obj_bid].ravel().copy()
        return np.concatenate([qp, qv, desired_orien])

    def set_env_state(self, state):
        qp = state[:30].copy()
        qv = state[30:60].copy()
        desired_orien = state[60:64].copy()
        self.set_state(qp, qv)
        self.model.body_quat[self.target_obj_bid] = desired_orien
        self.sim.forward()

    def set_env_full_state(self, full_state):
        self.set_env_state(full_state[:64])

    def mj_viewer_setup(self):
        self.viewer.cam.azimuth = -45
        self.sim.forward()
        self.viewer.cam.distance = 1.0

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if pen within 15 degrees of target for 20 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 20:
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
