import numpy as np
from sklearn.cluster import DBSCAN as skDBSCAN

class SKDBSCAN():
	def __init__(self, eps, min_samples):
		self.db = skDBSCAN(eps=eps, min_samples=min_samples)
	
	def __call__(self, points):
		self.db = self.db.fit(points)
		core_samples_mask = np.zeros_like(self.db.labels_, dtype=bool)
		core_samples_mask[self.db.core_sample_indices_] = True
		labels = self.db.labels_

		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		n_noise_ = list(labels).count(-1)
		return labels