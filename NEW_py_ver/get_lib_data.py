import os
import numpy as np

def get_tsplib_problems(local_path):
    count = 0
    for file in os.listdir(local_path):
        if file.endswith('.tsp'):
            with open(local_path+file, 'r') as f:
                data = f.readlines()
                if data[5] == 'NODE_COORD_SECTION\n' and data[4] == 'EDGE_WEIGHT_TYPE : EUC_2D\n':
                    print(file)
                    point_data = []
                    new_data = []
                    for d in data[6:]:
                        if d !='EOF\n' and d!='\n':
                            new_data.append(d)
                    data = new_data
                    for d in data:
                        d = d.strip().split(' ')
                        new_d = []
                        for point in d:
                            if point != '':
                                new_d.append(float(point))
                        point_data.append(new_d[1:])
                    point_data = np.array(point_data)
                    x = point_data
                    min_x = np.min(x)
                    max_x = np.max(x)
                    point_data = (x-np.min(x))/(np.max(x)-np.min(x))  # 最值归一化
                    print(point_data)
                    save_data = {'min_x':min_x,'max_x':max_x,'data':point_data}
                    npy_file = 'tsplib_npy/'+file+'.npy'
                    #print(npy_file)
                    np.save(npy_file, save_data)
                    count +=1
                    #break


def get_cvrplib_problems(local_path):
    count = 0
    with open(local_path, 'r') as f:
        files = f.readlines()
    for file in files:
        file = file.strip()
        if file.endswith('.vrp'):
            with open(file, 'r') as f:
                data = f.readlines()
                # print(data)
                # print(data[4])
                # print(data[4] == 'EDGE_WEIGHT_TYPE : EUC_2D\n')
                # print(data[6])
                # print(data[6] == 'NODE_COORD_SECTION\n')
                # if data[6] == 'NODE_COORD_SECTION\n' and data[4] == 'EDGE_WEIGHT_TYPE : EUC_2D\n':
                print(file)
                print('----------')
                if 'EUC_2D' in data[4] and 'NODE_COORD_SECTION' in data[6]:
                    print('euc 2d file')
                    print(file)
                    print('--------')
                    point_data = []
                    new_data = []
                    type = 0
                    node = []
                    demand = []
                    capacity = int(data[5].split('CAPACITY : ')[1])
                    for d in data[7:]:

                        # print(type)
                        # print(type, d=='DEMAND_SECTION\n')
                        # d = d.replace('\t0', ' ')
                        if 'DEMAND_SECTION' in d:
                            type = 1
                            continue
                        elif 'DEPOT_SECTION' in d:
                            # print('error')
                            break
                        if type == 0:
                            # print('d', d)
                            d = d.replace('\t', ' ')
                            d = d.strip().split(' ')
                            # print('d', d)
                            node.append(np.array(d[1:]).astype(np.float))
                        else:
                            d = d.replace('\t', ' ')
                            d = d.strip().split(' ')
                            # print(d)
                            # print(d[-1])
                            # print('split', d[-1].split("\t") )
                            demand.append([int(d[-1])])
                    # print(node, demand)
                    # data = new_data
                    # for d in data:
                    #     d = d.strip().split(' ')
                    #     new_d = []
                    #     for point in d:
                    #         if point != '':
                    #             new_d.append(float(point))
                    #     point_data.append(new_d[1:])
                    node = np.array(node)
                    x = node
                    # print(node)
                    node = (x - np.min(x)) / (np.max(x) - np.min(x))  # 最值归一化
                    min_x = np.min(x)
                    max_x = np.max(x)
                    # print(point_data)
                    demand = np.array(demand)
                    # k = (9-1)/(np.max(demand)-np.min(demand))
                    # demand = (1+k*(demand-np.min(demand))).astype(np.int)
                    # demand = (((demand-np.min(demand))/(np.max(demand)-np.min(demand)))*10).astype(np.int)

                    data = {'node': node, 'demand': demand, 'capacity': capacity, 'min_x': min_x, 'max_x': max_x}
                    # print('fsd')
                    # print(demand)
                    npy_file = 'cvrplib_npy/' + file.split('/')[-1] + '.npy'
                    # print(npy_file)
                    np.save(npy_file, data)
                    count += 1
                    # break