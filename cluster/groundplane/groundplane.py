import numpy as np

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