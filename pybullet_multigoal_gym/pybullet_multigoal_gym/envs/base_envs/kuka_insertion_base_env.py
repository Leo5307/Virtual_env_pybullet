import os
import numpy as np
import quaternion as quat
from pybullet_multigoal_gym.envs.base_envs.base_env import BaseBulletMGEnv
from pybullet_multigoal_gym.robots.kuka import Kuka
from pybullet_multigoal_gym.envs.base_envs.compute_reward import Compute_reward,Basic_compute_reward,Pick_up_reward,Reach_reward,Insert_reward
RANGE_TOLERANCE = 0.15
DISTANCE_THRESHOLD = 0.05
LOWER_BOUND = -15
UPPER_BOUND = 0.0

class KukaBulletInsertionEnv(BaseBulletMGEnv):
    """
    Base class for the insertion tasks with a Kuka iiwa 14 robot
    """

    def __init__(self, render=True, binary_reward=True,
                 image_observation=False, goal_image=False, depth_image=False, pcd=False, state_noise=False,
                 visualize_target=True, regenerate_goal_when_step=False,
                 manipulated_object_keys=None, goal_object_key='rectangle', orientation_informed_goal=False,
                 camera_setup=None, observation_cam_id=None, goal_cam_id=0, pcd_cam_id=0,
                 gripper_type='parallel_jaw', obj_range=RANGE_TOLERANCE, target_range=RANGE_TOLERANCE,
                 end_effector_start_on_table=False,
                 distance_threshold=DISTANCE_THRESHOLD, grasping=False):
        if observation_cam_id is None:
            observation_cam_id = [0]
        if manipulated_object_keys is None:
            manipulated_object_keys = ['rectangle', 'slot']
        self.binary_reward = binary_reward
        self.image_observation = image_observation
        self.goal_image = goal_image
        self.render_pcd = pcd
        if depth_image:
            self.render_mode = 'rgbd_array'
        else:
            self.render_mode = 'rgb_array'
        self.state_noise = state_noise
        self.visualize_target = visualize_target
        self.observation_cam_id = observation_cam_id
        self.goal_cam_id = goal_cam_id
        self.pcd_cam_id = pcd_cam_id
        self.regenerate_goal_when_step = regenerate_goal_when_step

        self.distance_threshold = distance_threshold
        self.grasping = grasping
        self.obj_range = obj_range
        self.target_range = target_range

        self.object_assets_path = os.path.join(os.path.dirname(__file__),
                                               "..", "..", "assets", "objects", "insertion")
        self.objects_urdf_loaded = False
        self.object_bodies = {
            'workspace': None,
            'slot': None,
            'rectangle': None,
            'target': None
        }
        self.object_initial_pos = {
            'workspace': [-0.58, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0],
            'slot': [-0.50, 0.0, 0.03, 0.0, 0.0, 0.0, 1.0],
            'rectangle': [-0.50, 0.0, 0.06, 0.0, 0.0, 0.0, 1.0],
            'target': [-0.55, 0.0, 0.03, 0.0, 0.0, 0.0, 1.0]
        }
        self.manipulated_object_keys = manipulated_object_keys
        self.goal_object_key = goal_object_key
        self.orientation_informed_goal = orientation_informed_goal

        self.desired_goal = None
        self.desired_goal_image = None

        robot = Kuka(grasping=grasping,
                     joint_control=False, end_effector_rotation_control=True,
                     gripper_type=gripper_type, end_effector_force_sensor=True,
                     end_effector_start_on_table=end_effector_start_on_table, table_surface_z=0.04,
                     obj_range=self.obj_range, target_range=self.target_range)

        BaseBulletMGEnv.__init__(self, robot=robot, render=render,
                                 image_observation=image_observation, goal_image=goal_image,
                                 camera_setup=camera_setup,
                                 seed=0, timestep=0.002, frame_skip=20)
    def _load_object(self):
                self.objects_urdf_loaded = True
                self.object_bodies['workspace'] = self._p.loadURDF(
                    os.path.join(self.object_assets_path, "workspace.urdf"),
                    basePosition=self.object_initial_pos['workspace'][:3],
                    baseOrientation=self.object_initial_pos['workspace'][3:])

                for object_key in self.manipulated_object_keys:
                    self.object_bodies[object_key] = self._p.loadURDF(
                        os.path.join(self.object_assets_path, object_key+".urdf"),
                        basePosition=self.object_initial_pos[object_key][:3],
                        baseOrientation=self.object_initial_pos[object_key][3:])

                self.object_bodies[self.goal_object_key+'_target'] = self._p.loadURDF(
                    os.path.join(self.object_assets_path, self.goal_object_key+"_target.urdf"),
                    basePosition=self.object_initial_pos['target'][:3],
                    baseOrientation=self.object_initial_pos['target'][3:])
            
    def _randomize_object_positions(self):

        # randomize object positions
        object_poses = []
        object_quats = []
        for object_key in self.manipulated_object_keys:
            done = False
            while not done:
                new_object_xy = self.np_random.uniform(self.robot.object_bound_lower[:-1],
                                                    self.robot.object_bound_upper[:-1])
                object_not_overlap = []
                for pos in object_poses + [self.robot.end_effector_tip_initial_position]:
                    object_not_overlap.append((np.linalg.norm(new_object_xy - pos[:-1]) > 0.06))
                if all(object_not_overlap):
                    object_poses.append(np.concatenate((new_object_xy.copy(), [self.object_initial_pos[object_key][2]])))
                    done = True

            orientation_euler = quat.as_euler_angles(quat.as_quat_array([1., 0., 0., 0.]))
            orientation_euler[-1] = self.np_random.uniform(-1.0, 1.0) * np.pi
            orientation_quat_new = quat.as_float_array(quat.from_euler_angles(orientation_euler))
            orientation_quat_new = np.concatenate([orientation_quat_new[1:], [orientation_quat_new[0]]], axis=-1)
            object_quats.append(orientation_quat_new.copy())

            self.set_object_pose(self.object_bodies[object_key],
                                object_poses[-1],
                                orientation_quat_new)
            
    def _task_reset(self, test=False):
        if not self.objects_urdf_loaded:
            # don't reload object urdf
            self._load_object()
            if not self.visualize_target:
                self.set_object_pose(self.object_bodies[self.goal_object_key+'_target'],
                                    [0.0, 0.0, -3.0],
                                    self.object_initial_pos['target'][3:])
        self._randomize_object_positions()
        # generate goals & images
        self._generate_goal()
        if self.goal_image:
            self._generate_goal_image()

    def _generate_goal(self):
        raise NotImplementedError()

    def _generate_goal_image(self):
        raise NotImplementedError()

    def _step_callback(self):
        pass
    
    def _calculate_all_state(self):
        state = []
        object_state = []
        achieved_goal = []
        gripper_xyz, gripper_rpy, gripper_finger_closeness, gripper_vel_xyz, gripper_vel_rpy, gripper_finger_vel, joint_poses, ee_joint_fx = self.robot.calc_robot_state()

        state = np.concatenate([gripper_xyz, gripper_rpy, gripper_vel_xyz, gripper_vel_rpy, gripper_finger_vel, joint_poses, ee_joint_fx])
        policy_state = np.concatenate([gripper_xyz, gripper_rpy, joint_poses, ee_joint_fx])
        if self.state_noise:
            # a 0.05 meters Gaussian noise on the state
            policy_state += self.np_random.standard_normal(size=policy_state.size) * 0.05
        state = np.concatenate([state, gripper_finger_closeness])
        policy_state = np.concatenate([policy_state, gripper_finger_closeness])

        for object_key in self.manipulated_object_keys:
            # object state: (x, y, z), (a, b, c, w)
            obj_xyz, (a, b, c, w) = self._p.getBasePositionAndOrientation(self.object_bodies[object_key])
            obj_euler = quat.as_euler_angles(quat.as_quat_array([w, a, b, c]))
            object_state.append(obj_xyz)
            object_state.append(obj_euler)
            if object_key == self.goal_object_key:
                achieved_goal.append(obj_xyz)
                if self.orientation_informed_goal:
                    achieved_goal.append(obj_euler)

        auxiliary_task_state = np.concatenate(object_state)
        achieved_goal = np.concatenate(achieved_goal)
        return state,policy_state,auxiliary_task_state,achieved_goal
    
    def _image_observation_handle(self,obs_dict):
        images = []
        state = obs_dict['observation']
        for cam_id in self.observation_cam_id:
            images.append(self.render(mode=self.render_mode, camera_id=cam_id))
        obs_dict['observation'] = images[0].copy()
        obs_dict['images'] = images
        obs_dict.update({'state': state.copy()})

        if self.goal_image:
            achieved_goal_img = self.render(mode=self.render_mode, camera_id=self.goal_cam_id)
            obs_dict.update({
                'achieved_goal_img': achieved_goal_img.copy(),
                'desired_goal_img': self.desired_goal_image.copy(),
            })

        if self.render_pcd:
            pcd = self.render(mode='pcd', camera_id=self.pcd_cam_id)
            obs_dict.update({'pcd': pcd.copy()})
        return obs_dict
    def _get_obs(self):
        # re-generate goals & images
        if self.regenerate_goal_when_step:
            self._generate_goal()
            if self.goal_image:
                self._generate_goal_image()
                
        assert self.desired_goal is not None
        gripper_xyz, _, _, _, _, _, _,_ = self.robot.calc_robot_state()
        state,policy_state,auxiliary_task_state,achieved_goal = self._calculate_all_state()
        
        assert achieved_goal.shape == self.desired_goal.shape

        obs_dict = {
            'observation': state.copy(),
            'policy_state': policy_state.copy(),
            'auxiliary_task_state': auxiliary_task_state.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.desired_goal.copy(),
            'subtask_rewards': self._compute_subtask_reward(gripper_xyz)
        }

        if self.image_observation:
            updated_obs_dict = self._image_observation_handle(obs_dict=obs_dict)

        return updated_obs_dict

    def _compute_subtask_reward(self, gripper_xyz):
        goal_object_xyz, (a, b, c, w) = self._p.getBasePositionAndOrientation(self.object_bodies['rectangle'])
        goal_object_euler = quat.as_euler_angles(quat.as_quat_array([w, a, b, c]))
        grasp_target_xyz, _, _, _, _, _ = self._p.getLinkState(self.object_bodies['rectangle'], 0)
        grasp_target_xyz = np.array(grasp_target_xyz)
        slot_target_xyz, (a, b, c, w), _, _, _, _ = self._p.getLinkState(self.object_bodies['slot'], 3)
        slot_target_euler = quat.as_euler_angles(quat.as_quat_array([w, a, b, c]))
        slot_target_xyz = np.array(slot_target_xyz)
        reach_target_xyz = slot_target_xyz.copy()
        reach_target_xyz[-1] += 2*RANGE_TOLERANCE
        insert_target_xyz = slot_target_xyz.copy()
        insert_target_xyz[-1] += RANGE_TOLERANCE
        
        pick_up_reward_calculator = Pick_up_reward(self.distance_threshold)
        reach_reward_calculator = Reach_reward(self.distance_threshold)
        insert_reward_calculator = Insert_reward(self.distance_threshold)
        
        reward_pick_up,achieved_pick_up = pick_up_reward_calculator.compute_reward(gripper_xyz,grasp_target_xyz,goal_object_xyz)
        reward_reach,achieved_reach =  reach_reward_calculator.compute_reward(reach_target_xyz,goal_object_xyz,slot_target_euler,goal_object_euler)
        reward_insert, achieved_insert =  insert_reward_calculator.compute_reward(insert_target_xyz,goal_object_xyz,slot_target_euler,goal_object_euler)
        
        return {
            'pick_up': np.clip(reward_pick_up, LOWER_BOUND, UPPER_BOUND),
            'pick_up_done': achieved_pick_up,
            'pick_up_desired_goal': np.concatenate([grasp_target_xyz, [RANGE_TOLERANCE]]),
            'pick_up_achieved_goal': np.concatenate([gripper_xyz, [goal_object_xyz[-1]]]),
            'reach': np.clip(reward_reach, LOWER_BOUND, UPPER_BOUND),
            'reach_done': achieved_reach,
            'reach_desired_goal': np.concatenate([reach_target_xyz, slot_target_euler]),
            'reach_achieved_goal': np.concatenate([goal_object_xyz, goal_object_euler]),
            'insert': np.clip(reward_insert, LOWER_BOUND, UPPER_BOUND),
            'insert_done': achieved_insert,
            'insert_desired_goal': np.concatenate([insert_target_xyz, slot_target_euler]),
            'insert_achieved_goal': np.concatenate([goal_object_xyz, goal_object_euler])
        }

    def _compute_reward(self, achieved_goal, desired_goal):
        # this computes the extrinsic reward
        computer_reward = Basic_compute_reward(distance_threshold = self.distance_threshold)
        return computer_reward.compute_reward(achieved_goal=achieved_goal,desired_goal=desired_goal,binary_reward = self.binary_reward)

    def set_object_pose(self, body_id, position, orientation=None):
        if orientation is None:
            orientation = self.object_initial_pos['workspace'][3:]
        self._p.resetBasePositionAndOrientation(body_id, position, orientation)
