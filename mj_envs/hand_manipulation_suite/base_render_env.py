import mujoco_py

DEFAULT_SIZE = 128


class RenderEnv:
    def __init__(self):
        self.viewer = None
        self._viewers = {}

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
                camera_name = 'track'

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
