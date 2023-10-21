#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:34:28 2023

@author: dreamone
"""

import pandas as pd
import numpy as np
import csv
import datetime
import scipy.stats as stats
import os
import sys
import time as systime


class Network:
    def __init__(self, node_num=0, edge_num=0, name=''):
        self.name = name
        self.node_num = node_num
        self.edge_num = edge_num
        self.matrix = np.full((self.node_num, self.node_num), -1, dtype=int)  # Adjacency matrix
        self.index = np.full((self.edge_num, 2), -1, dtype=int)  # Start and end of the edge
        self.distrib = np.zeros((edge_num, 2))  # MU and SIGMA of the edge
        self.test_od = None  # OD pairs for test

    # Initial the network by the CSV data
    def get_init(self, nu=0.4):
        file_net_path = os.getcwd() + f'/Networks/{self.name}/{self.name}_network.csv'
        net_data = pd.read_csv(file_net_path).values
        if len(net_data) != self.edge_num:
            print('Edge number is NOT equal to the lenth of Data !!!!!!!!!!!')
            sys.exit()
        else:
            # Numbered from 1, but stored from 0
            bias = 0 if min(net_data[:, 0]) == 0 or min(net_data[:, 1]) == 0 else 1
            for i in range(len(net_data)):
                self.matrix[int(net_data[i][0]) - bias, int(net_data[i][1]) - bias] = i
                self.index[i] = [int(net_data[i][0]) - bias, int(net_data[i][1]) - bias]
            self.distrib[:, 0] = net_data[:, 3]
            # Sigma is generated randomly based on the NU
            file_sigma = os.getcwd() + f'/Networks/{self.name}/{self.name}_{nu}_random_sigma.npy'
            if os.path.exists(file_sigma):
                sigma = np.load(file_sigma)
                print('Sigma file exists!')
            else:
                print('Sigma file do not exist，and sigma will be generated!')
                sigma = self.distrib[:, 0] * (np.random.rand(len(self.distrib[:, 0])) * nu)
                np.save(file_sigma, sigma)
            self.distrib[:, 1] = sigma
        # Test OD pairs load by CSV data
        file_od_path = os.getcwd() + f'/Networks/{self.name}/{self.name}_OD.csv'
        od_data = pd.read_csv(file_od_path).values
        self.test_od = od_data[:, 1:3].astype(int).tolist()


class LET:
    def __init__(self, network, origin, destination):
        self.cost_all, self.sigma_all, self.path = self._get_shortest_path(network, origin, destination)
        self.mu = network.distrib[self.path, 0].sum().item()
        self.sigma = np.sqrt((network.distrib[self.path, 1] ** 2).sum()).item()
        self.cost = self.cost_all[origin]
        self.path_print = [x + 1 for x in self.path]

    # Calculate the theoretical probability by MU and SIGMA, and calculate the actual probability by sampling
    def get_sota_prob(self, budget=1, epoch=1000):
        prob_theoretical = stats.norm.cdf(self.cost * budget, loc=self.mu, scale=self.sigma)
        file_observ = os.getcwd() + f'/Networks/{network.name}/{network.name}_{epoch}_random_observation.csv'
        if os.path.exists(file_observ):
            observ_epoch = np.loadtxt(file_observ, delimiter=",")
        else:
            print('Observation file do not exist，and observation will be generated randomly!')
            observ_epoch = np.zeros((epoch, network.edge_num))
            for ob_idx in range(epoch):
                observ_epoch[ob_idx] = np.random.normal(network.distrib[:, 0], network.distrib[:, 1],
                                                        size=len(network.distrib))
            np.savetxt(file_observ, observ_epoch, delimiter=",")
        prob_actual = ((observ_epoch[:, self.path].sum(1) < self.cost * budget).sum() / observ_epoch.shape[0])
        return prob_theoretical, prob_actual

    # Bellman-Ford algorithms for solving shortest paths
    def _get_shortest_path(self, network, origin, destination):
        # Reverse the direction of the edge
        revers_idx = np.zeros_like(network.index, dtype=int)
        revers_idx[:, 0] = network.index[:, 1]
        revers_idx[:, 1] = network.index[:, 0]
        # Solve shortest distance by Bellman-Ford algorithms
        distance = np.full((network.node_num,), np.inf)
        distance[destination] = 0
        variation = True
        while variation is True:
            add = network.distrib[:, 0] + distance[revers_idx[:, 0]]
            min_idx = add < distance[revers_idx[:, 1]]
            if variation in min_idx:
                distance[revers_idx[min_idx, 1]] = add[min_idx]
            else:
                variation = False
        # Solve shortest path by distance
        trace = [[] for _ in range(network.node_num)]
        dist = np.empty_like(distance)
        dist[:] = distance[:]
        min_trace = Destination
        for i in range(len(distance) - 1):
            dist[min_trace] = float('inf')
            min_dist_idx = np.argmin(dist)
            neig_idx = network.index[:, 0] == min_dist_idx
            neig_dist = (distance[network.index[neig_idx, 1]] + network.distrib[neig_idx, 0]) == distance[min_dist_idx]
            path = np.arange(network.edge_num)[neig_idx][neig_dist]
            if len(path) != 1:
                path = path[0]
            trace[min_dist_idx] = [_ for _ in trace[network.index[path, 1].item()]]
            trace[min_dist_idx].insert(0, path.item())
            min_trace = min_dist_idx
        sigma = [np.sqrt((network.distrib[x, 1] ** 2).sum()) if x != [] else 0 for x in trace]
        return distance, sigma, trace[origin]


def get_train(network, policy, origin, destination, Time, batch):
    # Prevent overflow
    def get_overflow_mask(data):
        limit = 700  # exp(double-float) can handle exponentials typically in the range of -709 to 709
        index = data > limit
        data[index] = limit
        index = data < -(limit)
        data[index] = -(limit)
        return data
    # Average over multiple samples
    delta = np.zeros_like(policy)
    for batch_idx in range(batch):
        observ = np.random.normal(network.distrib[:, 0], network.distrib[:, 1], size=len(network.distrib))
        time, path = 0, origin
        Prob, Hot = np.zeros(network.edge_num), np.zeros(network.edge_num)
        Observ, Remain_Time = np.zeros(network.edge_num), np.zeros(network.edge_num)
        Observ[:] = observ[:]
        # A trajectory to destination
        while time <= Time and path != destination:
            neig_idx = network.index[network.index[:, 0] == path, 1]
            path_idx = network.matrix[path, neig_idx]
            # Determine next steps by sampling
            theta1, theta2, ob = policy[path_idx, 0], policy[path_idx, 1], observ[path_idx]
            remain_time = Time - time
            exponent = get_overflow_mask(theta2 * (remain_time - ob) - theta1)
            p = np.exp(exponent) / np.exp(exponent).sum()
            next_idx = path_idx[np.where(np.random.rand() <= np.cumsum(p))[0][0]]

            time += observ[next_idx]
            path = np.where(next_idx == network.matrix[path, :])[0][0]
            observ[path_idx] = np.random.normal(network.distrib[path_idx, 0], network.distrib[path_idx, 1])
            Remain_Time[path_idx] = remain_time
            Prob[path_idx] = p
            Hot[path_idx] = 0
            Hot[next_idx] = 1
        if time <= Time and path == destination:
            r = 1
        else:
            r = 0
        delta[:, 0] += (r - 1 / 10) * ((-1) * (Hot - Prob))
        delta[:, 1] += (r - 1 / 10) * (Remain_Time - Observ) * (Hot - Prob)
    return delta / batch


def get_test(network, policy, origin, destination, Time, epoch):
    file_observ = os.getcwd() + f'/Networks/{network.name}/{network.name}_{epoch}_random_observation.csv'
    observ_epoch = np.loadtxt(file_observ, delimiter=",")
    count = 0
    start = systime.time()

    for i in range(epoch):
        observ = observ_epoch[i]
        time, path = 0, origin

        while time <= Time and path != destination:
            neig_idx = network.index[network.index[:, 0] == path, 1]
            path_idx = network.matrix[path, neig_idx]
            theta1, theta2, ob = policy[path_idx, 0], policy[path_idx, 1], observ[path_idx]
            remain_time = Time - time

            next_idx = path_idx[np.argmax(theta2 * (remain_time - ob) - theta1)]
            time += observ[next_idx]
            path = np.where(next_idx == network.matrix[path, :])[0][0].item()
            observ[path_idx] = np.random.normal(network.distrib[path_idx, 0], network.distrib[path_idx, 1])
        if time <= Time and path == destination:
            count += 1

    cost_time = systime.time() - start
    return count / epoch, cost_time / epoch


def get_run(network, Origin, Destination, let, alpha, budget, Epoch, Epoch_test, batch):
    policy = np.zeros((network.edge_num, 2))
    let.sigma_all[Destination] = min(network.distrib[:, 1])
    policy[:, 1] = 1 / np.array(let.sigma_all)[network.index[:, 1]]
    policy[:, 0] = let.cost_all[network.index[:, 1]] * policy[:, 1]
    Time = let.cost * budget
    accuracy_record = [0] * 12
    cost_time_record = [0] * 12

    print('*' * 3, ' Test before Train', '*' * 3)
    accuracy_record[0], cost_time_record[0] = get_test(network, policy, Origin, Destination, Time, Epoch_test)
    print(f'Test Times:{Epoch_test}   Accuracy:{(accuracy_record[0] * 100):.2f}%')
    
    print('*' * 3, ' Train ', '*' * 3)
    for e in range(10):
        for i in range(Epoch // (10 * batch)):
            delta = get_train(network, policy, Origin, Destination, Time, batch)
            policy += alpha * delta
        accuracy_record[e + 1], cost_time_record[e + 1] = get_test(network, policy, Origin, Destination, Time,
                                                                    Epoch_test)
        print(f'Epoch Times:{(e + 1) * Epoch / 10}   Accuracy:{(accuracy_record[e + 1] * 100):.2f}%')

    print('*' * 3, ' Test after Train', '*' * 3)
    accuracy_record[-1], cost_time_record[-1] = get_test(network, policy, Origin, Destination, Time, Epoch_test)
    cost_time = sum(cost_time_record) / 12
    print(f'Test Times:{Epoch_test}   Accuracy:{(accuracy_record[-1] * 100):.2f}%')
    print(f'The Cost Time is {cost_time:.2e}s')

    now = datetime.datetime.now()
    print(f'{now.month}/{now.day}  {now.hour}:{now.minute} {now.second}s\n')
    return accuracy_record, cost_time


# main funtion
print('*' * 20, ' START ', '*' * 20)
Net_list = ['SiouxFalls', 'Friedrichshain', 'Anaheim', 'Winnipeg', 'Chicago_Sketch']
Net_node_edge_num_list = [[24, 76], [224, 531], [416, 914], [1052, 2836], [933, 2950]]
Budget_list = [0.95, 1, 1.05]

alpha = 1e-6
Epoch = 100000
batch = 100
Epoch_test = 5000

for net_idx in range(len(Net_list)):
    net = Net_list[net_idx]
    print('\n', '*' * 15, f' Now The Net is :{net}', '*' * 15)

    net_node_edge_num = Net_node_edge_num_list[net_idx]
    Node, Edge = net_node_edge_num[0], net_node_edge_num[1]
    network = Network(Node, Edge, net)
    network.get_init(nu=0.4)
    OD_list = network.test_od
    print(
        f'{network.name}:: Node Number:{network.node_num}   Edge Number:{network.edge_num}   Test OD pairs Number:{len(OD_list)}')

    t = datetime.datetime.now()
    prefix = os.getcwd() + f'/Networks/{net}/GPG-RT_Record/{t.month}_{t.day}_{t.hour}_{t.minute}'
    file_name0 = prefix + '_LET_record.csv'
    file_name1 = prefix + '_param_record.csv'
    file_name2 = prefix + '_train_record.csv'
    with open(file_name0, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['No.', 'Orgin', 'Destination', 'LET Cost', 'LET Path', 'LET mu', 'LET sigma', 'Budget',
                  'Theoretical Target', 'Actual Target']
        csvwriter.writerow(header)
    with open(file_name1, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['No.', 'Orgin', 'Destination', 'Budget', 'Actual Target', 'Alpha', 'Epoch', 'TAbT', 'TAaT', 'Best',
                  'Best index', 'Cost Time']
        csvwriter.writerow(header)
    with open(file_name2, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['No.', 'TAbT', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'TAaT']
        csvwriter.writerow(header)

    count_condition, count_test = 0, 0
    for od_idx in range(0, len(OD_list)):
        od = OD_list[od_idx]
        Origin, Destination = od[0] - 1, od[1] - 1
        let = LET(network, Origin, Destination)

        print('\n',
              f'Condition {od_idx + 1} / {len(OD_list)}:: Network:{network.name}   Origin:{od[0]}   Destination:{od[1]}')
        print(f'Priori:: OD:{od[0]}-{od[1]}   LET:{let.cost}   LET Path:{let.path_print}\n')

        for bg_idx in range(len(Budget_list)):
            budget = Budget_list[bg_idx]
            print('*' * 10, f'Now Budget is {budget}', '*' * 10)

            target_theoretical, target_actual = let.get_sota_prob(budget, Epoch_test)
            count_condition += 1
            let_record = [count_condition, od[0], od[1], let.cost, let.path_print, let.mu, let.sigma, budget,
                          target_theoretical, target_actual]
            LET_record = [f'{x}' for x in let_record[0: 5]] + [f'{x:.4f}' for x in let_record[5:]]
            with open(file_name0, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(LET_record)
            print(f'Budget:{budget}  Theoretical Probability:{(target_theoretical * 100):.2f}%  '
                  f'Actual Probability:{(target_actual * 100):.2f}%')

            record, cost_time = get_run(network, Origin, Destination, let, alpha, budget, Epoch, Epoch_test, batch)
            count_test += 1
            best = max(record[1: -1])
            best_idx = record[1: -1].index(best) + 1
            train_record = [f'{count_test}'] + [f'{x:.4f}' for x in record]
            param_list = [target_actual, alpha, Epoch, record[0], record[-1], best, best_idx, cost_time]
            param_record = [f'{count_test}', f'{od[0]}', f'{od[1]}', f'{budget}']
            param_record += [f'{param_list[0]:.4f}', f'{param_list[1]:.2e}', f'{param_list[2]}']
            param_record += [f'{x:.4f}' for x in param_list[3: 6]]
            param_record += [f'{param_list[-2]}'] + [f'{param_list[-1]:.4e}']
            with open(file_name1, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(param_record)
            with open(file_name2, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(train_record)

print('*' * 20, ' OVER! ', '*' * 20)
