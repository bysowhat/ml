import numpy as np
from dbscan import SKDBSCAN
from cluster.data.base import Data
from ground.groundplane import GroundPlaneSimple


def main():
	lidar_dir = 'C:\\Users\\yu.bai\\Datasets\\Kitti\\training\\velodyne'
	lidar_type = 'bin'
	datas = Data(lidar_dir, lidar_type)
	#xmin,xmax,ymin,ymax,zmin,zmax
	pc_range = [-20, 20, -20, 20, -1.3, 2.0]
	gp = GroundPlaneSimple(pc_range)
	dbs = SKDBSCAN(eps=0.5, min_samples=4)
	for points in datas:
		points = gp.filter_points(points)
		labels = dbs(points)
		# datas.show(points)
		# datas.show_labels(points, labels)
		datas.show_convechull(points, labels)

if __name__ == '__main__':
	main()