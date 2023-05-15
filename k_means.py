import numpy as np
import matplotlib.pyplot as plt
import random


class Point:

    def __init__(self, x, y, cluster=-1):
        self.x = x
        self.y = y
        self.cluster = cluster

    def euclidean_distance(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def chebyshev_distance(self, other):
        return max(abs(self.x - other.x), abs(self.y - other.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.cluster == other.cluster

    def __ne__(self, other):
        return not self.__eq__(other)


class ClusteriserKMeans:

    def __init__(self):
        np.random.seed = 42
        self.points = []
        self.curr_means = []
        self.cluster_number = -1
        self.output = ''

    def add_point(self, x, y):
        self.points.append(Point(x, y))

    def get_data_x_y_for_cluster(self, cl_num):
        x_cl = []
        y_cl = []
        for point in self.points:
            if point.cluster == cl_num:
                x_cl.append(point.x)
                y_cl.append(point.y)
        return x_cl, y_cl

    def get_mean_for_cluster(self, cluster):
        x_cl = []
        y_cl = []
        for point in self.points:
            if point.cluster == cluster:
                x_cl.append(point.x)
                y_cl.append(point.y)
        return np.mean(x_cl), np.mean(y_cl)

    def update_euclidean_distance(self):
        for point in self.points:
            distances = [point.euclidean_distance(self.curr_means[i])
                         for i in range(self.cluster_number)]
            point.cluster = np.argmin(distances)

    def update_chebyshev_distance(self):
        for point in self.points:
            distances = [point.chebyshev_distance(self.curr_means[i])
                         for i in range(self.cluster_number)]
            point.cluster = np.argmin(distances)

    def show_current_state(self, start=False):
        markers = ['o', 'v', 'D', 's']
        fig, ax = plt.subplots(figsize=(4, 4))
        if start:
            print('start!')
            x_cl, y_cl = self.get_data_x_y_for_cluster(-1)
            ax.scatter(x_cl, y_cl, marker='o', label='points')
        else:
            for cluster in range(self.cluster_number):
                x_cl, y_cl = self.get_data_x_y_for_cluster(cluster)
                ax.scatter(x_cl, y_cl, marker=markers[cluster], label='cluster ' + str(cluster))
        means_x, means_y = get_data_x_y_of_means(self.curr_means)
        ax.scatter(means_x, means_y, marker="d", label='cluster centers', c='red')
        ax.grid()
        ax.legend()

    def get_clusters(self, cl_num=2, dist='euclidean'):
        self.cluster_number = cl_num

        cluster_idx = list(range(len(self.points)))
        cluster_idx = random.sample(cluster_idx, k=self.cluster_number)

        for idx in cluster_idx:
            self.curr_means.append(self.points[idx])

        iteration = 0
        self.show_current_state(True)
        plt.savefig('report' + str(iteration) + '.png')
        while True:
            prev_means = self.curr_means.copy()

            if dist == 'euclidean':
                self.update_euclidean_distance()
            else:
                self.update_chebyshev_distance()

            self.output += f'\nIteration #{iteration}\nCurrent state:\n'
            for p in self.points:
                self.output += f'{p.x}, {p.y}, {p.cluster}\n'

            for cluster in range(self.cluster_number):
                means_x, means_y = self.get_mean_for_cluster(cluster)
                self.curr_means[cluster] = Point(means_x, means_y)
                self.output += f'Means value for cluster {cluster}: {means_x}, {means_y}\n'

            if prev_means == self.curr_means:
                break
            iteration += 1
            self.show_current_state()
            plt.savefig('report' + str(iteration) + '.png')

        return iteration, self.output

    def clear_clusters(self):
        for point in self.points:
            point.cluster = -1
        self.curr_means = []
        self.cluster_number = -1
        self.output = ''


def get_data_x_y_of_means(means):
    x_cl = []
    y_cl = []
    for point in means:
        x_cl.append(point.x)
        y_cl.append(point.y)
    return x_cl, y_cl
