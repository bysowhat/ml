import numpy as np
from sklearn.decomposition import PCA

class GroundPlaneSimple():
    def __init__(self, pc_range):
        self.pc_range = pc_range
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = self.pc_range

    def filter_points(self, points):
        mask = np.ones(points.shape[0], dtype=np.bool_)
        mask = np.logical_and(mask, points[:, 0] > self.xmin)
        mask = np.logical_and(mask, points[:, 0] < self.xmax)
        mask = np.logical_and(mask, points[:, 1] > self.ymin)
        mask = np.logical_and(mask, points[:, 1] < self.ymax)
        mask = np.logical_and(mask, points[:, 2] > self.zmin)
        mask = np.logical_and(mask, points[:, 2] < self.zmax)
        return points[mask]

class GroundPlane(GroundPlaneSimple):
    def __init__(self, pc_range):
        super().__init__(pc_range)
        self.pca = PCA(n_components=3)

    def compute_plane(self, points):
        '''
        Args:
            points: (n, 3), n is the total num of points.
        Returns:
            [a,b,c,d]: plane equation: ax+by+cz+d=0
        '''
        self.pca.fit(points)
        sv = self.pca.singular_values_
        comp = self.pca.components_
        min_index = np.argmin(sv)
        norm = comp[min_index]
        d = -sum(points[0]*norm)
        return list(norm) + [d]


if __name__ == '__main__':
    from cluster.data.base import Data

    lidar_dir = 'C:\\Users\\yu.bai\\Datasets\\Kitti\\training\\velodyne'
    lidar_type = 'bin'
    datas = Data(lidar_dir, lidar_type)
    pc_range = [-20, 20, -20, 20, -5, 3.0]

    gp = GroundPlane(pc_range)
    for points in datas:
        points = gp.filter_points(points)
        points = np.asarray([[-10.0, 2.0,1.0], [2.0, 12.0,-1.0], [2.0, -3.0, -2.0]])
        plane = gp.compute_plane(points)
        datas.show_points_on_ground(points, plane)