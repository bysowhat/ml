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
    def __init__(self, 
                 pc_range, 
                 split_num=3,
                 max_iter=3, 
                 n_lpr=200, 
                 th_seed=0.4, 
                 th_dis=0.2, 
                 hmin=-2.2, 
                 hmax=-1.4,
                 stop_min = 0.1,
                 datas=None):
        super().__init__(pc_range)
        self.split_num = split_num
        self.pca = PCA(n_components=3)
        self.n_lpr = n_lpr
        self.th_seed = th_seed
        self.max_iter = max_iter
        self.th_dis = th_dis
        self.hmin = hmin
        self.hmax = hmax
        self.datas = datas
        self.stop_min = stop_min

    @staticmethod
    def points_2_plane_dis(points, plane):
        plane = np.asarray(plane)
        dis = (np.dot(points, plane[:3]) + plane[3])/np.linalg.norm(plane[:3])
        return dis

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
        mask0 = points[:,2] > self.hmin
        mask1 = points[:,2] < self.hmax
        points = points[np.logical_and(mask0, mask1)]
        # self.datas.show(points)
        hs = np.sort(points[:,2])
        averg_h = np.mean(hs[:self.n_lpr])
        mask = points[:,2]<averg_h + self.th_seed
        # self.datas.show(points[mask])
        return points[mask]
    
    def points_2_plane_abs_dis(self, points, plane):
        return np.abs(self.points_2_plane_dis(points, plane))
    
    def split_points(self, points):
        step = (self.xmax - self.xmin)/self.split_num
        points_list = []
        self.x_ths = []
        for i in range(self.split_num):
            mask1 = points[:, 0] > (self.xmin + step*i)
            mask2 = points[:, 0] < (self.xmin + step*(i+1))
            mask = np.logical_and(mask1, mask2)
            points_list.append(points[mask])
            self.x_ths.append((self.xmin + step*i, self.xmin + step*(i+1)))
        # for p in points_list:
        #     self.datas.show(p)
        return points_list
    
    def get_mask(self, dis, bins=10, th_pt=0.5):
        min = np.min(dis)
        max = np.max(dis)
        bin = (max-min)/bins
        sum_num = dis.shape[0]
        num_list = []
        mask_list = []
        for i in range(bins):
            mask0 = dis > min + i*bin
            mask1 = dis <= min + (i+1)*bin
            num_list.append(np.sum(mask0*mask1))
            mask_list.append(np.logical_and(mask0, mask1))
        
        results = []
        min_start = 0
        min_end = 0
        min_l = 9999
        for i in range(bins):
            pt = 0.0
            for j in range(i, bins, 1):
                pt += num_list[j]/sum_num
                if pt > th_pt:
                    results.append((i, j))
                    if j-i < min_l:
                        min_end = j
                        min_start = i
                        min_l = j - i           

        mask = mask_list[min_start]
        for i in range(min_start, min_end+1, 1):
            mask = np.logical_or(mask, mask_list[i])
        return mask
    
    def _segment_ground(self, points):
        seeds = self.get_seeds(points)
        print('start iter')
        for _ in range(self.max_iter):
            plane = self.compute_plane(seeds)
            # self.datas.show_points_on_ground(seeds, plane)
            dis = self.points_2_plane_dis(points, plane)
            mask = np.abs(dis) < self.th_dis
            dis = dis[mask]
            seeds = points[mask]
            print(np.mean(dis))
            if np.abs(np.mean(dis)) < self.stop_min:
                break
            mask = self.get_mask(dis)
            seeds = seeds[mask]
            # self.datas.show_points_on_ground(seeds, plane)
        return plane

    def segment_ground(self, points):
        points_list = self.split_points(points)
        plane_list = []
        for ps in points_list:
            # plane_list.append(self._segment_ground(ps))
            plane_list.append(self._segment_ground(points_list[1]))
        return plane_list, self.x_ths

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

        gp = GroundPlane(pc_range,max_iter=4, n_lpr=200, th_seed=0.6, th_dis=0.6, hmin = -2.2, hmax = -1.4, stop_min=0.1, datas=datas)
        for i,points in enumerate(datas):
            if i < 1:
                continue
            points = gp.filter_points(points)
            # datas.show(points)
            plane_list, x_ths = gp.segment_ground(points)
            datas.show_ground_segmentation(points, plane_list, x_ths, th=0.0)
    
    test_segment_ground()