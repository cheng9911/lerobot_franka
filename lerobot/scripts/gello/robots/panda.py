import time
from typing import Dict

import numpy as np

from gello.robots.robot import Robot
import grpc

MAX_OPEN = 0.0889


class PandaRobot(Robot):
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "127.0.0.1"):
        from polymetis import GripperInterface, RobotInterface

        self.robot = RobotInterface(
            ip_address=robot_ip,
        )
        self.gripper = GripperInterface(
            ip_address="192.168.1.108",
        )
        self.robot.go_home()
        self.robot.start_joint_impedance()
        self.gripper.goto(width=MAX_OPEN, speed=255, force=25)
        self.controller_active = True
        time.sleep(1)

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        return 8

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = self.robot.get_joint_positions()
        gripper_pos = self.gripper.get_state()
        pos = np.append(robot_joints, gripper_pos.width / MAX_OPEN)
        return pos

    # def command_joint_state(self, joint_state: np.ndarray) -> None:
    #     """Command the leader robot to a given state.

    #     Args:
    #         joint_state (np.ndarray): The state to command the leader robot to.
    #     """
    #     import torch
    #     if not self.robot.is_running_policy():
    #         print("[INFO] Starting joint impedance controller...")
    #         try:
    #             self.robot.start_joint_impedance()
    #             self.gripper.goto(width=MAX_OPEN, speed=255, force=25)
    #             time.sleep(0.04)
    #             # print("[INFO] Controller started.")
    #         except Exception as e:
    #             print("[ERROR] Failed to start joint impedance:", e)
       
            

    #     self.robot.update_desired_joint_positions(torch.tensor(joint_state[:-1]))
    #     self.gripper.goto(width=(MAX_OPEN * (1 - joint_state[-1])), speed=200, force=2)
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        import torch
        if not self.controller_active:
            print("[INFO] Starting joint impedance controller...")
            try:
                self.robot.start_joint_impedance()
                self.controller_active = True
                self.gripper.goto(width=MAX_OPEN, speed=255, force=25)
                time.sleep(0.04)
            except Exception as e:
                print("[ERROR] Failed to start joint impedance:", e)
                self.controller_active = False

        try:
            self.robot.update_desired_joint_positions(torch.tensor(joint_state[:-1]))
        except grpc.RpcError as e:
            print("[WARN] Controller likely lost. ")
            
            self.controller_active = False
            
        try:
            grip_width = MAX_OPEN * (1 - joint_state[-1])
            grip_width = max(0.0, min(MAX_OPEN, grip_width))
            self.gripper.goto(width=grip_width, speed=200, force=100)
        except Exception as e:
            print("[ERROR] Gripper command failed:", e)
        # self.gripper.goto(width=(MAX_OPEN * (1 - joint_state[-1])), speed=200, force=2)

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]])
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }


def main():
    robot = PandaRobot()
    current_joints = robot.get_joint_state()
    print("Current joints:", current_joints)
    # move a small delta 0.1 rad
    move_joints = current_joints + 0.05
    # make last joint (gripper) closed
    move_joints[-1] = 0.5
    time.sleep(1)
    m = 0.09
    robot.gripper.goto(1 * m, speed=255, force=255)
    time.sleep(1)
    robot.gripper.goto(0 * m, speed=255, force=255)
    time.sleep(1)
    robot.gripper.goto(1.1 * m, speed=255, force=255)
    time.sleep(1)


if __name__ == "__main__":
    main()
