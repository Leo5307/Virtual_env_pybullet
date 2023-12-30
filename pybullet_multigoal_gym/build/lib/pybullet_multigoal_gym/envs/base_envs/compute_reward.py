import numpy as np
import quaternion as quat

class Compute_reward:
    def __init__(self,distance_threshold):
        self.distance_threshold = distance_threshold
        # print("test")
        #  pass 
    
    def basic_compute_reward(self, achieved_goal, desired_goal,binary_reward):
        # this computes the extrinsic reward
        assert achieved_goal.shape == desired_goal.shape
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        not_achieved = (d > self.distance_threshold)
        if binary_reward:
            return -not_achieved.astype(np.float32), ~not_achieved
        else:
            return -d, ~not_achieved
        
    def pick_up_reward(self,gripper_xyz,grasp_target_xyz,goal_object_xyz):
        # print("1")
        # pick-up reward
        # this reward specifies grasping point & goal object height
        d_goal_obj_grip = np.linalg.norm(grasp_target_xyz - gripper_xyz, axis=-1) + np.abs(0.15 - goal_object_xyz[-1])
        # this reward only specifies goal object height
        # d_goal_obj_grip = np.abs(0.15 - goal_object_xyz[-1])
        reward_pick_up = -d_goal_obj_grip
        achieved_pick_up = (d_goal_obj_grip < self.distance_threshold)
        return reward_pick_up,achieved_pick_up
        
    def reach_reward(self,reach_target_xyz,goal_object_xyz,slot_target_euler,goal_object_euler):
        # print("2")
    # reach reward
    # reach_target_xyz = slot_target_xyz.copy()
    # reach_target_xyz[-1] += 0.06
        d_goal_obj_reach_slot = np.linalg.norm(goal_object_xyz - reach_target_xyz, axis=-1) + \
                                np.linalg.norm(goal_object_euler - slot_target_euler.copy(), axis=-1)
        reward_reach = -d_goal_obj_reach_slot
        achieved_reach = (d_goal_obj_reach_slot < self.distance_threshold)
        return reward_reach,achieved_reach
        
    def insert_reward(self,insert_target_xyz,goal_object_xyz,slot_target_euler,goal_object_euler):
        # print("3")
    # insert reward
        # insert_target_xyz = slot_target_xyz.copy()
        # insert_target_xyz[-1] += 0.03
        d_goal_obj_insert_slot = np.linalg.norm(goal_object_xyz - insert_target_xyz, axis=-1) + \
                                np.linalg.norm(goal_object_euler - slot_target_euler.copy(), axis=-1)
        reward_insert = -d_goal_obj_insert_slot
        achieved_insert = (d_goal_obj_insert_slot < self.distance_threshold)
        return reward_insert,achieved_insert