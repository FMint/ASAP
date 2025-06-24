import numpy as np

# 替换为你的文件路径
# data_list 是一个列表，每个元素对应一个学习迭代的数据
file_path = "/root/ASAP_opti/logs/G1_23dof_MotionTracking/20250624_122807-test_for_data_collection-motion_tracking-g1_23dof/test_for_data_collection_full_raw_data.npy"
loaded_data_array = np.load(file_path, allow_pickle=True)
# if loaded_data_array.ndim == 0:
#     data_list = loaded_data_array.item()
# # 如果 loaded_data_array 是一个 1 维数组，且只有一个元素（这个元素就是你的 Python 列表）
# elif loaded_data_array.ndim == 1 and len(loaded_data_array) == 1:
#     data_list = loaded_data_array[0]
# # 其他情况，可能是保存方式有误，或者你期望的数据结构更复杂
# else:
#     raise ValueError("Unexpected .npy file structure. Expected a scalar array or a 1-element array containing the data list.")

data_list = loaded_data_array

print(f"Total iterations saved: {len(data_list)}")

# 访问第 0 个迭代的数据
first_iteration_data = data_list[0] 
print(f"Keys in first iteration data: {first_iteration_data.keys()}") # 应该看到 env_0, env_1 等

# 访问第 0 个迭代中，环境 0 的 DOF 位置数据
env0_dof_pos_in_first_iter = first_iteration_data['env_0']['dof_pos']
print(f"Shape of env_0 dof_pos in first iteration: {env0_dof_pos_in_first_iter.shape}") # (num_steps_per_env, 23)
print(f"First step's DOF Pos for Env 0: {env0_dof_pos_in_first_iter[0]}")

# 访问第 5 个迭代中，环境 1 的动作数据
fifth_iteration_data = data_list[4] # 索引从0开始，所以是data_list[4]
env1_actions_in_fifth_iter = fifth_iteration_data['env_1']['actions']
print(f"Shape of env_1 actions in fifth iteration: {env1_actions_in_fifth_iter.shape}")
print(f"First step's Actions for Env 1 in fifth iteration: {env1_actions_in_fifth_iter[0]}")