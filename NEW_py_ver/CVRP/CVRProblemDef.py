
import torch
import numpy as np
torch.manual_seed(1)

def get_random_problems(batch_size, problem_size):
    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 40:
        demand_scaler = 35
    elif problem_size >= 50:
        demand_scaler = 30 + int(problem_size / 5)
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)
    return depot_xy, node_xy, node_demand


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data


def get_local_problem(local_path):
    data = np.load(local_path, allow_pickle=True).item()

    node_xy = list(data['node'])
    depot_xy = node_xy[0]
    node_demand = list(data['demand'])
    capacity = data['capacity']
    min_x = data['min_x']
    max_x = data['max_x']
    current_len = len(node_xy)
    problem_size = current_len

    if current_len < 100:
        for i in range(100 - current_len):
            node_xy.append(node_xy[0])
            node_demand.append(node_demand[0])
    node_xy = np.array(node_xy)
    depot_xy = np.array(depot_xy)
    node_demand = np.array(node_demand)
    depot_xy = torch.tensor(depot_xy, dtype=torch.float).unsqueeze(0).unsqueeze(0)
    node_xy = torch.tensor(node_xy, dtype=torch.float).unsqueeze(0)
    node_demand = torch.tensor(node_demand, dtype=torch.float).permute(1, 0) / capacity
    return depot_xy, node_xy, node_demand, min_x, max_x
