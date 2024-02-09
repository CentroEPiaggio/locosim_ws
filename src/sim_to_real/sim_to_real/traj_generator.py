from rclpy.node import Node
from sensor_msgs.msg import JointState
from pi3hat_moteus_int_msgs.msg import JointsCommand,JointsStates
import rosbag2_py
from rclpy.serialization import serialize_message
import time 
import torch
import rclpy
import cmath

from datetime import datetime

# lf, lh, rf, rh

class TrajectoryGenerator:

    def __init__(self, num_envs=1, num_dof=8, device='cpu', episode_length_s=5., task='stand_up', hz=1., n=4, amp=0.05, z=-0.33):

        """
        :param num_envs: The number of robots, for the real robot must be 1
        :param num_dof: The number of actuated joints of the robot, for Mulinex must be 8
        :param device: Whether to use cpu or gpu, for the real robot use cpu
        :param episode_length_s: The length of the experiment
        :param task: The name of the task, one of ['stand_up', 'push_ups', 'pitching', 'rolling', 'trotting_in_place']
        :param hz: Frequency pitching(1), rolling(1) and trotting_in_place(5) (suggested values)
        :param n: Used for task=push_ups only, the number of push-ups.
        """

        self._tasks = ['stand_up', 'push_ups', 'pitching', 'rolling', 'trotting_in_place', 'moving_along_x', 'circle']
        if task not in self._tasks:
            raise ValueError(f'Unknown task "{task}"! Task value must be in {self._tasks}')

        self._num_envs = num_envs
        self._num_dof = num_dof
        self._device = device
        self._episode_length_s = episode_length_s
        self._task = task
        self._hz = torch.tensor(hz)
        self._n = torch.tensor(n)
        self._amp = amp
        self._z_feet = z
        self._a1 = 0.19
        self._a2 = 0.19

        if task in ['stand_up', 'push_ups']:
            # default_dof_pos_REST
            self._default_dof_pos = torch.tensor([[3.141592653589793, -3.141592653589793, -3.141592653589793, 3.141592653589793, -3.141592653589793, 3.141592653589793, 3.141592653589793, -3.141592653589793]] * self._num_envs, dtype=torch.float, device=self._device)
        else:
            # DEFAULT_dof_pos
            self._default_dof_pos = torch.tensor([[2.0944, -2.0944, -2.0944, 2.0944, -1.0472, 1.0472, 1.0472, -1.0472]] * self._num_envs, dtype=torch.float, device=self._device)

        self._dof_pos_cmd = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)

        self._x = torch.zeros((self._num_envs, 4), dtype=torch.float, device=self._device)
        self._z = torch.zeros((self._num_envs, 4), dtype=torch.float, device=self._device)

        self._q = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self._prev_q = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self._q_out_cs = torch.zeros((self._num_envs, self._num_dof), dtype=torch.bool, device=self._device)
        self.publish_flag = True

    def get_dof_pos_cmd(self, t):
        if self._task == 'stand_up':
            self._set_dof_pos_cmd_for_stand_up(t)  # Requires "default_dof_pos_REST" and base_init_pos_z = 0.04
        if self._task == 'push_ups':
            self._set_dof_pos_cmd_for_n_push_ups(self._n, t)  # Requires "default_dof_pos_REST" and base_init_pos_z = 0.04
        if self._task == 'pitching':
            self._set_dof_pos_cmd_for_pitching_at_hz(self._hz, t)  # Requires "DEFAULT_dof_pos" and base_init_pos_z = 0.33
        if self._task == 'rolling':
            self._set_dof_pos_cmd_for_rolling_at_hz(self._hz, t)  # Requires "DEFAULT_dof_pos" and base_init_pos_z = 0.33
        if self._task == 'trotting_in_place':
            self._set_dof_pos_cmd_for_trotting_in_place_at_hz(self._hz, t)  # Requires "DEFAULT_dof_pos" and base_init_pos_z = 0.33
        if self._task == 'moving_along_x':
           self._set_dof_pos_cmd_for_moving_along_x(t)
        if self._task == 'circle':
           self._set_dof_pos_cmd_for_circle(t)  # Requires base_init_pos_z = 0.2
        return self._dof_pos_cmd.squeeze().clone()

    def _set_dof_pos_cmd_for_stand_up(self, t):
        if t > 0:
            lf_foot_z, lh_foot_z, rf_foot_z, rh_foot_z = [- 0.33 / self._episode_length_s * t] * 4
            self._dof_pos_cmd[:] = self._get_dof_pos_cmd_for_feet_pos(lf_foot_z=lf_foot_z, lh_foot_z=lh_foot_z, rf_foot_z=rf_foot_z, rh_foot_z=rh_foot_z)
        else:
            self._dof_pos_cmd[:] = self._default_dof_pos

    def _set_dof_pos_cmd_for_n_push_ups(self, n, t):
        if t not in [i * self._episode_length_s / n for i in range(n)]:
            lf_foot_z, lh_foot_z, rf_foot_z, rh_foot_z = [- 0.33 * torch.sin(2 * torch.pi * n / 10 * t).abs()] * 4
            self._dof_pos_cmd[:] = self._get_dof_pos_cmd_for_feet_pos(lf_foot_z=lf_foot_z, lh_foot_z=lh_foot_z, rf_foot_z=rf_foot_z, rh_foot_z=rh_foot_z)
        else:
            self._dof_pos_cmd[:] = self._default_dof_pos

    def _set_dof_pos_cmd_for_pitching_at_hz(self, hz, t):
        lf_foot_z, rf_foot_z = [- 0.33 + 0.03 * torch.sin(2 * torch.pi * hz * t)] * 2
        lh_foot_z, rh_foot_z = [- 0.33 - 0.03 * torch.sin(2 * torch.pi * hz * t)] * 2
        self._dof_pos_cmd[:] = self._get_dof_pos_cmd_for_feet_pos(lf_foot_z=lf_foot_z, lh_foot_z=lh_foot_z, rf_foot_z=rf_foot_z, rh_foot_z=rh_foot_z)

    def _set_dof_pos_cmd_for_rolling_at_hz(self, hz, t):
        lf_foot_z, lh_foot_z = [- 0.33 + 0.02 * torch.sin(2 * torch.pi * hz * t)] * 2
        rf_foot_z, rh_foot_z = [- 0.33 - 0.02 * torch.sin(2 * torch.pi * hz * t)] * 2
        self._dof_pos_cmd[:] = self._get_dof_pos_cmd_for_feet_pos(lf_foot_z=lf_foot_z, lh_foot_z=lh_foot_z, rf_foot_z=rf_foot_z, rh_foot_z=rh_foot_z)

    def _set_dof_pos_cmd_for_trotting_in_place_at_hz(self, hz, t):
        sin_1 = 0.05 * torch.sin(2 * torch.pi * hz * t)
        sin_2 = 0.05 * torch.sin(2 * torch.pi * hz * t + torch.pi)
        sin_1 *= sin_1 > 0
        sin_2 *= sin_2 > 0
        lf_foot_z, rh_foot_z = [- 0.33 + sin_1] * 2
        rf_foot_z, lh_foot_z = [- 0.33 + sin_2] * 2
        self._dof_pos_cmd[:] = self._get_dof_pos_cmd_for_feet_pos(lf_foot_z=lf_foot_z, lh_foot_z=lh_foot_z, rf_foot_z=rf_foot_z, rh_foot_z=rh_foot_z)
    def _set_dof_pos_cmd_for_moving_along_x(self, t):
       sin = self._amp * torch.sin(2 * torch.pi * self._hz * t)
       self._dof_pos_cmd[:] = self._get_dof_pos_cmd_for_feet_pos(lf_foot_x=sin, lh_foot_x=sin, rf_foot_x=sin, rh_foot_x=sin,
                                                                 lf_foot_z=self._z_feet, lh_foot_z=self._z_feet, rf_foot_z=self._z_feet, rh_foot_z=self._z_feet)

    def _set_dof_pos_cmd_for_circle(self, t):
       sin_1 = self._amp * torch.sin(2 * torch.pi * self._hz * t)
       sin_2 = - 0.28 + 0.08 * torch.sin(2 * torch.pi * self._hz * t + torch.pi / 2)
       self._dof_pos_cmd[:] = self._get_dof_pos_cmd_for_feet_pos(lf_foot_x=sin_1, lh_foot_x=sin_1, rf_foot_x=sin_1, rh_foot_x=sin_1,
                                                                 lf_foot_z=sin_2, lh_foot_z=sin_2, rf_foot_z=sin_2, rh_foot_z=sin_2)


    def _inv_kin(self):

        self._q[:, -4:] = torch.acos((torch.square(self._x) + torch.square(self._z) - self._a1 ** 2 - self._a2 ** 2) / (2 * self._a1 * self._a2))

        self._q[:, 0] = torch.atan2(- self._z[:, 0], self._x[:, 0]) + torch.atan2(self._a2 * torch.sin(self._q[:, 4]), self._a1 + self._a2 * torch.cos(self._q[:, 4]))
        self._q[:, 1] = torch.atan2(self._z[:, 1], - self._x[:, 1]) - torch.atan2(self._a2 * torch.sin(self._q[:, 5]), self._a1 + self._a2 * torch.cos(self._q[:, 5]))
        self._q[:, 2] = torch.atan2(self._z[:, 2], self._x[:, 2]) - torch.atan2(self._a2 * torch.sin(self._q[:, 6]), self._a1 + self._a2 * torch.cos(self._q[:, 6]))
        self._q[:, 3] = torch.atan2(- self._z[:, 3], - self._x[:, 3]) + torch.atan2(self._a2 * torch.sin(self._q[:, 7]), self._a1 + self._a2 * torch.cos(self._q[:, 7]))

        self._q[:, [4, 7]] *= -1.

        self._q[self._q > torch.pi] -= 2 * torch.pi
        self._q[self._q < - torch.pi] += 2 * torch.pi

        self._q_out_cs[:] = (torch.sqrt(torch.square(self._x) + torch.square(self._z)) > (self._a1 + self._a2)).repeat(1, self._num_dof // 4)
        self._q[self._q_out_cs] = self._prev_q[self._q_out_cs]

        self._prev_q[:] = self._q

    def _get_dof_pos_cmd_for_feet_pos(self, *, lf_foot_x=0, lh_foot_x=0, rf_foot_x=0, rh_foot_x=0, lf_foot_z=0, lh_foot_z=0, rf_foot_z=0, rh_foot_z=0):
        self._x[:, 0] = lf_foot_x
        self._x[:, 1] = lh_foot_x
        self._x[:, 2] = rf_foot_x
        self._x[:, 3] = rh_foot_x
        self._z[:, 0] = lf_foot_z
        self._z[:, 1] = lh_foot_z
        self._z[:, 2] = rf_foot_z
        self._z[:, 3] = rh_foot_z
        self._inv_kin()
        return self._q

# LF_HFE, LH_HFE, RF_HFE, RH_HFE, LF_KFE, LH_KFE, RF_KFE, RH_KFE
class ROS_Trajectory_Generator(Node):

    def __init__(self,hz = 1, episode_length_s=1,task='pitching',n=2,debug=False,homing_time=5.0,bag_folder="/home/jacopo/Desktop/RL_EXP/",amp=0.05,z=-0.33):
        super().__init__("ros_trajectory_generator")
        self.generator = TrajectoryGenerator(
            episode_length_s=episode_length_s,
            task=task,
            hz=hz,
            n=n,
            amp=amp,
            z=z)
        self.offset = 0.0
        self.homing_os = 0.0
        self.degub = debug
        self.ep_len = episode_length_s
        self.pub_flag = True
        if(self.degub == True):
            self.msg_cmd = JointState()
            self.pub = self.create_publisher(JointState,"joint_states",10)
        else:
            self.msg_cmd = JointsCommand()
            self.pub = self.create_publisher(JointsCommand,"joint_controller/command",10)

        
        self.first_pos = self.generator.get_dof_pos_cmd(0.0).tolist()
        self.joint_list = ["LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE", "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE"]
        self.jnt_list = ["RF_HFE","RF_KFE","LF_HFE","LF_KFE","LH_HFE","LH_KFE","RH_HFE","RH_KFE"]

        self.homing_time = homing_time

        self.writer = rosbag2_py.SequentialWriter()

        bag_now = datetime.now()
        storage_options = rosbag2_py._storage.StorageOptions(
            uri=bag_folder+ 'exp_RL' +  bag_now.strftime("%m_%d_%Y_%H_%M_%S"),
            storage_id='sqlite3')
            #  storage_id='mcap')
        converter_options = rosbag2_py._storage.ConverterOptions('', '')
        self.writer.open(storage_options, converter_options)

        topic_info = rosbag2_py._storage.TopicMetadata(
            name='command',
            type='pi3hat_moteus_int_msgs/msg/JointsCommand',
            serialization_format='cdr')
        
        

        self.writer.create_topic(topic_info)

        topic_info_1 = rosbag2_py._storage.TopicMetadata(
            name='state',
            type='pi3hat_moteus_int_msgs/msg/JointsStates',
            serialization_format='cdr')
        
        self.writer.create_topic(topic_info_1)

        self.sub = self.create_subscription(
            JointsStates,
            "state_broadcaster/joints_state",
            self.state_callback,
            10
        )

        self.timer = self.create_timer(timer_period_sec=0.005,callback=self.timer_callback)

        self.state = 0

    def rosnow_2_sec(self,sec,nsec):
        return float(sec) + float(nsec)/pow(10,9)
    
    def reshape_jnt_val(self,unord_val):
        ord_val = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        ord_val[0] = unord_val[2]
        ord_val[1] = unord_val[6]
        ord_val[2] = unord_val[0]
        ord_val[3] = unord_val[4]
        ord_val[4] = unord_val[1]
        ord_val[5] = unord_val[5]
        ord_val[6] = unord_val[3]
        ord_val[7] = unord_val[7]
        return ord_val
        

    def timer_callback(self):
        if(self.state == 0):
            # homing needed
            time_ros = self.get_clock().now()
            [s,ns] = time_ros.seconds_nanoseconds()
            now = self.rosnow_2_sec(s,ns)
            if(self.homing_os == 0.0):
                self.homing_os = now
            t = now - self.homing_os
            if(t >self.homing_time):
                if(t > self.homing_time + 10.0):
                    self.state = 1
                t = self.homing_time
            
            des_hom_pos = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            for i in range(len(des_hom_pos)):
                des_hom_pos[i] = (self.first_pos[i]/self.homing_time)*t
            self.msg_cmd.name = self.jnt_list
            self.msg_cmd.position = self.reshape_jnt_val(des_hom_pos)
            
            self.msg_cmd.velocity = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            self.msg_cmd.effort = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            if(self.degub == False):
                self.msg_cmd.kp_scale = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
                self.msg_cmd.kd_scale = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
            self.msg_cmd.header.stamp = time_ros.to_msg()
            self.pub_flag = True
            for i in self.msg_cmd.position:
                if cmath.isnan(i):
                    self.pub_flag = False
                    break
            if(self.pub_flag):
                self.pub.publish(self.msg_cmd)
            
        if(self.state == 1):
            
            time_ros = self.get_clock().now()
            [s,ns] = time_ros.seconds_nanoseconds()
            now = self.rosnow_2_sec(s,ns)
            if(self.offset==0.0):
                self.offset = now
                self.offset = self.offset 
            t = now - self.offset
            if(t > self.ep_len):
                t = self.ep_len
                self.state = 2
            # print(f"the comp time is {t}")
            # print(self.generator.get_dof_pos_cmd(t))            
            self.msg_cmd.position = self.reshape_jnt_val(self.generator.get_dof_pos_cmd(t).tolist())
            self.msg_cmd.velocity = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            self.msg_cmd.effort = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            if(self.degub == False):
                self.msg_cmd.kp_scale = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
                self.msg_cmd.kd_scale = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
            self.msg_cmd.header.stamp = time_ros.to_msg()
            self.pub_flag = True
            for i in self.msg_cmd.position:
                if cmath.isnan(i):
                    self.pub_flag = False
                    break
            if(self.pub_flag):
                self.pub.publish(self.msg_cmd)
            
            self.writer.write(
            'command',
            serialize_message(self.msg_cmd),
            self.get_clock().now().nanoseconds)
        if(self.state == 2):
            print("end Task")
            self.timer.cancel()

    def state_callback(self, msg):
        self.writer.write(
            'state',
            serialize_message(msg),
            self.get_clock().now().nanoseconds)
            

def main(args=None):
    rclpy.init(args=args)
    RL_node = ROS_Trajectory_Generator(debug=False,homing_time=5,hz=5.0,episode_length_s=5, task="trotting_in_place",amp=0.1,z=-0.2)
    rclpy.spin(RL_node)
    RL_node.destroy_node()
    rclpy.shutdown()
    


if __name__ == '__main__':
    main()
