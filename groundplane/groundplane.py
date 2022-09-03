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
    def __init__(self, pc_range, max_iter=5, n_lpr=200, th_seed=0.4, th_dis=0.2):
        super().__init__(pc_range)
        self.pca = PCA(n_components=3)
        self.n_lpr = n_lpr
        self.th_seed = th_seed
        self.max_iter = max_iter
        self.th_dis = th_dis


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
    
    def get_seeds(self, points):
        mask0 = points[:,2]>-2.2
        mask1 = points[:,2]<-1.4
        points = points[np.logical_and(mask0, mask1)]
        hs = np.sort(points[:,2])
        averg_h = np.mean(hs[:self.n_lpr])
        mask = points[:,2]<averg_h + self.th_seed
        return points[mask]
    
    def points_2_plane_dis(self, points, plane):
        plane = np.asarray(plane)
        dis = np.abs(np.dot(points, plane[:3]) + plane[3])/np.linalg.norm(plane[:3])
        return dis
    
    def segment_ground(self, points):
        seeds = self.get_seeds(points)
        for _ in range(self.max_iter):
            plane = self.compute_plane(seeds)
            dis = self.points_2_plane_dis(points, plane)
            mask = dis < self.th_dis
            seeds = points[mask]
        return plane


if __name__ == '__main__':
    from cluster.data.base import Data

    def test_compute_plane():
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
    
    def test_segment_ground():
        lidar_dir = 'C:\\Users\\yu.bai\\Datasets\\Kitti\\training\\velodyne'
        lidar_type = 'bin'
        datas = Data(lidar_dir, lidar_type)
        pc_range = [-50, 50, -50, 50, -5, 3.0]
        # pc_range = [-50, 50, -50, 50, -2.2, -1.4]

        gp = GroundPlane(pc_range)
        for points in datas:
            points = gp.filter_points(points)
            # datas.show(points)
            plane = gp.segment_ground(points)
            datas.show_ground_segmentation(points, gp.points_2_plane_dis(points, plane), th=0.2)
    
    test_segment_ground()