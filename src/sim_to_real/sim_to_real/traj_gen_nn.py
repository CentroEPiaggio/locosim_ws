from rclpy.node import Node
from sensor_msgs.msg import JointState
from pi3hat_moteus_int_msgs.msg import JointsCommand,JointsStates
import rosbag2_py
from rclpy.serialization import serialize_message
import time 
import torch
import rclpy
import cmath
from .utils.rlg_utils import build_rlg_model
import yaml 
from datetime import datetime

# lf, lh, rf, rh

class NNTrajectoryGenerator:

    def __init__(self, model, num_envs=1, num_dof=8, device='cpu', model_device='cuda:0', homing_time=10., no_policy_time=5.,
                 policy_dt=.02, num_obs=36, default_dof_pos=(2.0944, -2.0944, -2.0944, 2.0944, -1.0472, 1.0472, 1.0472, -1.0472),
                 deterministic=True, lin_vel_scale=2., ang_vel_scale=.25, dof_pos_scale=1., dof_vel_scale=.05, action_scale=.5):

        self._model = model

        self._num_envs = num_envs
        self._num_dof = num_dof
        self._device = device
        self._model_device = model_device
        self._homing_time = homing_time
        self._no_policy_time = no_policy_time
        self._policy_dt = policy_dt
        self._num_obs = num_obs
        self._deterministic = deterministic

        self._default_dof_pos = torch.tensor([default_dof_pos] * self._num_envs, dtype=torch.float, device=self._device)
        self._dof_pos_cmd = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self._obs = torch.zeros((self._num_envs, self._num_obs), dtype=torch.float, device=self._device)

        self._ang_vel_scale = ang_vel_scale
        self._cmd_scale = torch.tensor([lin_vel_scale, ang_vel_scale], dtype=torch.float, device=self._device)
        self._dof_pos_scale = dof_pos_scale
        self._dof_vel_scale = dof_vel_scale
        self._action_scale = action_scale

        self._gravity_vec = torch.tensor([[0., 0., -1.]] * self._num_envs, dtype=torch.float, device=self._device)
        self._orientation = torch.tensor([[1., 0., 0., 0.]] * self._num_envs, dtype=torch.float, device=self._device)
        self._projected_gravity = self._quat_rotate_inverse(self._orientation, self._gravity_vec)
        self._ang_vel = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device)
        self._cmd = torch.zeros((self._num_envs, 2), dtype=torch.float, device=self._device)
        self._prev_dof_pos = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self._dof_pos = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self._dof_vel = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)
        self._actions = torch.zeros((self._num_envs, self._num_dof), dtype=torch.float, device=self._device)

        self._init_state()

    @staticmethod
    @torch.jit.script
    def _quat_rotate_inverse(q, v):
        shape = q.shape
        q_w = q[:, 0]
        q_vec = q[:, 1:]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c

    def _init_state(self):
        self._dof_pos[:] = self._default_dof_pos

    def _set_cmd(self, t, vx, wz):
        if t < self._homing_time:
            self._cmd[:] = torch.tensor([0., 0.], dtype=torch.float, device=self._device)
        else:
            self._cmd[:] = torch.tensor([vx, wz], dtype=torch.float, device=self._device)

    def _get_obs(self):
        self._obs[:] = torch.cat((self._projected_gravity,
                                  self._orientation,
                                  self._ang_vel * self._ang_vel_scale,
                                  self._cmd * self._cmd_scale,
                                  self._dof_pos * self._dof_pos_scale,
                                  self._dof_vel * self._dof_vel_scale,
                                  self._actions), dim=-1)

    def _run_inference(self):
        input_dict = {'is_train': False,
                      'prev_actions': None,
                      'obs': self._obs.to(self._model_device),
                      'rnn_states': None}

        with torch.no_grad():
            res_dict = self._model(input_dict)

        if self._deterministic:
            self._actions[:] = res_dict['mus']
        else:
            self._actions[:] = res_dict['actions']

    def _set_dof_pos_cmd(self, t):
        if t < self._no_policy_time:
            self._dof_pos_cmd[:] = self._default_dof_pos
        else:
            self._dof_pos_cmd[:] = self._action_scale * self._actions + self._default_dof_pos

    def _refresh_state(self):
        self._ang_vel[:, 2] = self._cmd[:, 1]
        self._prev_dof_pos[:] = self._dof_pos
        self._dof_pos[:] = self._dof_pos_cmd
        self._dof_vel[:] = (self._dof_pos - self._prev_dof_pos) / self._policy_dt

    def get_dof_pos_cmd(self, t, vx=0., wz=0.):
        self._set_cmd(t, vx, wz)
        self._get_obs()
        self._run_inference()
        self._set_dof_pos_cmd(t)
        self._refresh_state()
        return self._dof_pos_cmd.squeeze().clone()

# LF_HFE, LH_HFE, RF_HFE, RH_HFE, LF_KFE, LH_KFE, RF_KFE, RH_KFE
class ROS_NN_Trajectory_Generator(Node):

    def __init__(self,model,debug, ep_len, homing_time= 5.0, no_policy_time= 2.0,bag_folder="/home/jacopo"):
        super().__init__("ros_trajectory_generator")

        self.generator = NNTrajectoryGenerator(model,homing_time=homing_time,no_policy_time=no_policy_time)
        self.offset = 0.0
        self.homing_os = 0.0
        self.degub = debug
        self.ep_len = no_policy_time + homing_time + ep_len
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

        self.timer = self.create_timer(timer_period_sec=0.02,callback=self.timer_callback)

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
            self.msg_cmd.position = self.reshape_jnt_val(self.generator.get_dof_pos_cmd(t,0.5, 0.).tolist())
        #     print(f" the time is {t} and the reference pos are {self.msg_cmd.position}")
            self.msg_cmd.name = self.jnt_list
            self.msg_cmd.velocity = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            self.msg_cmd.effort = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            if(self.degub == False):
                self.msg_cmd.kp_scale = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
                self.msg_cmd.kd_scale = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
            self.msg_cmd.header.stamp = time_ros.to_msg()
            self.pub_flag = True
        #     for i in self.msg_cmd.position:
        #         if cmath.isnan(i):
        #             self.pub_flag = False
        #             break
        #     if(self.pub_flag):
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
    pth_path = "/home/jacopo/Documents/locosim_ws/src/rlg_quad_controller/models/Mulinex_2024_02_13-12_00_00/MulinexTerrainNew.pth"
    yaml_path = "/home/jacopo/Documents/locosim_ws/src/rlg_quad_controller/models/Mulinex_2024_02_13-12_00_00/config.yaml"
    with open(yaml_path,"r") as f:
        params = yaml.safe_load(f)
    model = build_rlg_model(pth_path,params)
    RL_node = ROS_NN_Trajectory_Generator(model=model,debug=False,ep_len= 5,homing_time=5.0,no_policy_time=2.0,bag_folder="/home/jacopo/Desktop/RL_EXP/NN_TRAJ")
    rclpy.spin(RL_node)
    RL_node.destroy_node()
    rclpy.shutdown()
    


if __name__ == '__main__':
    main()
