import numpy as np

file_path = "/root/ASAP_opti/logs/G1_23dof_MotionTracking/20250624_122807-test_for_data_collection-motion_tracking-g1_23dof/test_for_data_collection_full_raw_data.npy"
loaded_data_array = np.load(file_path, allow_pickle=True)

data_list = loaded_data_array

print(f'\nenv number: {len(data_list[0])}\n')
print(f"Total iterations saved: {len(data_list)}\n")

print(f"data type collected: {data_list[0]['env_0'].keys()}\n")

# # 访问第 0 个迭代的数据
# first_iteration_data = data_list[0] 
# print(f"number of envs: {first_iteration_data.keys()}\n") # 应该看到 env_0, env_1 等

# # 访问第 0 个迭代中，环境 0 的 DOF 位置数据
# env0_dof_pos_in_first_iter = first_iteration_data['env_0']['dof_pos']
# print(f"Steps' count of env_0 DOF-POS in iteration 0: {env0_dof_pos_in_first_iter.shape}\n") # (num_steps_per_env, 23)
# print(f"Env_0 DOF-POS in step 0, iteration 0: {env0_dof_pos_in_first_iter[0]}\n")

# # 访问第 5 个迭代中，环境 1 的动作数据
# fifth_iteration_data = data_list[4] # 索引从0开始，所以是data_list[4]
# env1_actions_in_fifth_iter = fifth_iteration_data['env_1']['actions']
# print(f"Steps' count of env_1 ACTIONS in iteration 4: {env1_actions_in_fifth_iter.shape}\n")
# print(f"Env_1 action in step 8, iteration 4: {env1_actions_in_fifth_iter[8]}\n")



chosen_env = 'env_0'  # 可以选择 'env_0', 'env_1', 等等
chosen_iteration = 10  # 可以选择 0, 1, 2, ..., 9, ...
chosen_step = 5
chosen_data_type = 'dof_pos' #'dof_pos', 'dof_vel', 'root_lin_vel', 'root_ang_vel', 
                        #'root_pos', 'root_rot', 'actions', 'terminate'

chosen_data = data_list[chosen_iteration][chosen_env]

print(f'Chosen iteration: {chosen_iteration}, chosen env: {chosen_env}\n')


chosen_data = data_list[chosen_iteration][chosen_env]


print(f'total steps of iteration : {len(chosen_data)}\n')
print(f'chosen step: {chosen_step}\n')

print(f'data type chosen: {chosen_data_type}\n')

print(chosen_data[chosen_data_type][chosen_step])
print('\n')
print(chosen_data[chosen_data_type][chosen_step].shape)
print('\n')



