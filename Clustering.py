#-*-encoding:utf-8-*-
from sklearn import cluster
from utils import GetPlaceName, GetGeocode, locationbox, latlon2km
import smopy
import time
import numpy as np
import json
import networkx as nx
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import mixture, preprocessing

#Set random seed
np.random.seed(0)

class Placename_clustering:

	def __init__(self,SnapIter,visualize=False):

		self.SnapIter = SnapIter
		self.Snap  = None
		self.visualize = visualize

	def build_clusters(self):

		self.labels = Placename_clustering.run(X=self.Snap['LOC'],place_names = self.Snap['PLACE'],visualize=self.visualize,Grid=self.SnapIter.Grid,name=self.Snap['TimeWindow'])

	def next(self):
		self.Snap = self.SnapIter.next()

	@staticmethod
	def run(X,place_names,visualize,Grid,name):

		place_set   = set(place_names)


		if visualize!=False:

			place2index = dict(zip(place_set,range(0,len(place_set))))
			index2place = dict(zip(range(0,len(place_set)),place_set))
			labels = [place2index[place] for place in place_names]

			#Presentation-related lists
			color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
			visual_patterns = ['ob','og','or','oc','om','oy','ok','ow','xb','xg','xr','xc','xm','xy','xk','xw']

			#Plot results
			map_ = smopy.Map([Grid[1],Grid[0],Grid[3],Grid[2]],z=13)
			mpl  = plt.subplot(1,1,1)
			mpl.set_xticks([])
			mpl.set_yticks([])
			mpl.grid(False)
			mpl.set_xlim(0, map_.w)
			mpl.set_ylim(map_.h, 0)
			mpl.axis('off')

			mpl  = map_.show_mpl(mpl)
			printed = []
			ClusterSize = Counter(labels)

			index2place = dict(zip(range(0,len(place_set)),place_set))

			for k in range(0,len(X)):

				x,y = map_.to_pixels(X[k][0],X[k][1])
				mpl.plot(x,y,'%s'%visual_patterns[labels[k]%len(visual_patterns)])
				if labels[k] not in printed:
					mpl.text(x, y, '%d at %s'%(ClusterSize[labels[k]],index2place[labels[k]]), fontsize=8,color=visual_patterns[labels[k]%len(visual_patterns)][1])
					printed.append(labels[k])

			plt.tight_layout()
			pp = PdfPages('PlaceName%s.pdf'%name)
			plt.savefig(pp, format='pdf')
			pp.close()

			return [index2place[index] for index in labels]

		else:
			return place_names



class GMM_clustering:

	def __init__(self,SnapIter,components=5,visualize=False):

		self.SnapIter = SnapIter
		self.Snap  = None
		self.components = components
		self.visualize = visualize

	def build_clusters(self):

		self.labels = GMM_clustering.run(np.vstack(self.Snap['LOC']),components = self.components,visualize=self.visualize,Grid=self.SnapIter.Grid,name=self.Snap['TimeWindow'])

	def next(self):
		self.Snap = self.SnapIter.next()

	@staticmethod
	def run(X,components,visualize,Grid,name):

		#Pre-processing function
		scaler = preprocessing.StandardScaler().fit(X)
		X_scaled = scaler.transform(X)

		#bayesian information criterion
		lowest_bic = np.infty
		bic = []
		if type(components) is int:
			n_components_range = range(1, components+1)
		elif type(components) is list:
			n_components_range = components
		else:
			raise TypeError('components type should be list or int not'%type(components))

		cv_types = ['spherical', 'tied', 'diag', 'full']

		for cv_type in cv_types:

			for n_components in n_components_range:

				# Fit a mixture of gaussians with EM
				gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)

				gmm.fit(X_scaled)
				bic.append(gmm.bic(X_scaled))

				if bic[-1] < lowest_bic:
					lowest_bic = bic[-1]
					best_gmm = gmm

		#Predictions of best result
		labels = best_gmm.predict(X_scaled)

		# Plot the BIC scores
		if visualize!=False:


			#Presentation-related lists
			color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
			visual_patterns = ['ob','og','or','oc','om','oy','ok','ow','xb','xg','xr','xc','xm','xy','xk','xw']

			fig = plt.figure(figsize=(40,20))
			gs  = gridspec.GridSpec(1,2,width_ratios=[2,3])
			spl = plt.subplot(gs[0])

			bic = np.array(bic)
			bars = []

			for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
				xpos = np.array(n_components_range) + .2 * (i - 2)
				bars.append(plt.bar(xpos, bic[i * len(n_components_range):
				                             (i + 1) * len(n_components_range)],
				                   width=.2, color=color))

			plt.xticks(n_components_range)
			plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
			plt.title('BIC score per model')
			xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
			    .2 * np.floor(bic.argmin() / len(n_components_range))
			plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=30)
			spl.set_xlabel('Number of components')
			spl.legend([b[0] for b in bars], cv_types)

			#Plot best result of GMM
			map_ = smopy.Map([Grid[1],Grid[0],Grid[3],Grid[2]],z=13)
			mpl  = plt.subplot(gs[1])
			mpl.set_xticks([])
			mpl.set_yticks([])
			mpl.grid(False)
			mpl.set_xlim(0, map_.w)
			mpl.set_ylim(map_.h, 0)
			mpl.axis('off')

			mpl  = map_.show_mpl(mpl)
			printed = []
			ClusterSize = Counter(labels)

			for k in range(0,len(X)):

				x,y = map_.to_pixels(X[k][0],X[k][1])
				mpl.plot(x,y,'%s'%visual_patterns[labels[k]])
				if labels[k] not in printed:
					mpl.text(x, y, '%d'%ClusterSize[labels[k]], fontsize=8,color=visual_patterns[labels[k]][1])
					printed.append(labels[k])

			plt.tight_layout()
			pp = PdfPages('GMM%s.pdf'%name)
			plt.savefig(pp, format='pdf')
			pp.close()

		return labels

class TweetGraph:

	def __init__(self,SnapIter,name='TweetGraph',edge_method='inverse'):

		self.SnapIter   = SnapIter
		self.Snap       = None
		self.Graph      = nx.Graph()
		self.Graph.name = name
		self.edge_method = edge_method

	def build_graph(self):

		if self.Snap!=None:

			assert len(self.Snap['LOC'])==len(self.Snap['TEXT']), 'No. of geotags (text) are not equal'
			for i in range(0,len(self.Snap['LOC'])):
				j = i + 1
				while j<len(self.Snap['LOC']):
					self.Graph.add_edge( i, j, {'weight':SnapGraph.edge_weight((self.Snap['LOC'][i]),(self.Snap['LOC'][j]),method=self.edge_method)} )
					j = j + 1

			#List of TEXT attrib in a list
			self.TEXT = self.Snap['TEXT']

		else:
			raise ValueError('A Snap (self.Snap) has not been initialized for conversion to graph')

	def next(self):
		self.Snap = SnapGraph.ProcessSnap(self.SnapIter.next())

	def get_text(self,node):
		return self.TEXT[node]

	@staticmethod
	def ProcessSnap(Snap):

		del_list = []

		for i in range(len(Snap['LOC'])):

			j = i + 1

			while j < len(Snap['LOC']):
				if Snap['LOC'][i]!=Snap['LOC'][j]:
					pass
				else:
					Snap['TEXT'][i]+= '\t__and__\t'+Snap['TEXT'][j]
					del_list.append(j)
				j+=1

		del_list = sorted(del_list)
		del_list.reverse()

		if len(del_list)>0:
			for item in del_list:

				del Snap['TEXT'][item]
				del Snap['LOC'][item]

		return Snap

	@staticmethod
	def min_cut(G):
		"""Process graph here to cut"""
		return [G1,G2]

	@staticmethod
	def centrality(G,method=None):
		"""Assign scores to node (centrality)"""
		if method==None:
			return G

	@staticmethod
	def edge_weight((lat1,lon1),(lat2,lon2),method='inverse'):

		distance = latlon2km((lat1,lon1),(lat2,lon2))

		if method=='inverse':
			return 1.0/distance
		else:
			raise ValueError('%s is not defined'%method)

class Base_clustering:

	def __init__(self,SnapIter,method,method_name,visualize=False):

		self.SnapIter  = SnapIter
		# Templeates :
		# 1. K-Means   = KMeans(n_clusters=n_clusters,init='random',n_init=10,n_jobs=2)
		self.method    = method
		self.visualize = visualize
		self.method_name = method_name

	def build_clusters(self):

		self.labels =  Base_clustering.run(X=np.vstack(self.Snap['LOC']),method=self.method,visualize=self.visualize,Grid=self.SnapIter.Grid,name=self.method_name+'__'+str(self.Snap['TimeWindow']))

	def next(self):
		self.Snap = self.SnapIter.next()

	@staticmethod
	def run(X,method,visualize,Grid,name):

		#Pre-processing function
		scaler = preprocessing.StandardScaler().fit(X)
		X_scaled = scaler.transform(X)

		labels = method.fit_predict(X_scaled)

		if visualize:

			map_ = smopy.Map([Grid[1],Grid[0],Grid[3],Grid[2]],z=13)
			visual_patterns = ['ob','og','or','oc','om','oy','ok','ow','xb','xg','xr','xc','xm','xy','xk','xw']



			 #Plot successive heirarchies
			fig = plt.figure(figsize=(10,10))

			mpl  = plt.subplot(1,1,1)
			mpl.set_xticks([])
			mpl.set_yticks([])
			mpl.grid(False)
			mpl.set_xlim(0, map_.w)
			mpl.set_ylim(map_.h, 0)
			mpl.axis('off')

			mpl  = map_.show_mpl(mpl)
			ClusterSize = Counter(labels)
			printed = []


			for k in range(0,len(X)):

				x,y = map_.to_pixels(X[k][0],X[k][1])
				mpl.plot(x,y,'%s'%visual_patterns[labels[k]%len(visual_patterns)])
				if labels[k] not in printed:
					mpl.text(x, y, '%d'%ClusterSize[labels[k]], fontsize=15,color=visual_patterns[labels[k]][1])
					printed.append(labels[k])



			plt.tight_layout()
			pp = PdfPages('%s.pdf'%name)
			plt.savefig(pp, format='pdf')
			pp.close()



		return labels

class Heirarichal_clustering:

	def __init__(self,SnapIter,visualize):

		self.SnapIter = SnapIter
		self.visualize = visualize

	def build_clusters(self):

		self.labels =  Heirarichal_clustering.run(np.vstack(self.Snap['LOC']),visualize=self.visualize,Grid=self.SnapIter.Grid,name=self.Snap['TimeWindow'])

	def next(self):
		self.Snap = self.SnapIter.next()

	@staticmethod
	def run(X,visualize,Grid,name):

		#Pre-processing function
		scaler = preprocessing.StandardScaler().fit(X)
		X_scaled = scaler.transform(X)

		labels = []
		n_clusters = 2

		while n_clusters<len(X):

			clf = cluster.Ward(n_clusters=n_clusters,compute_full_tree=True,memory='./')
			labels.append(clf.fit_predict(X_scaled))
			n_clusters +=1

		if visualize:

			map_ = smopy.Map([Grid[1],Grid[0],Grid[3],Grid[2]],z=13)
			visual_patterns = ['ob','og','or','oc','om','oy','ok','ow','xb','xg','xr','xc','xm','xy','xk','xw']
			ct = 0
			for lbls in labels :

				 #Plot successive heirarchies
				fig = plt.figure(figsize=(10,10))

				mpl  = plt.subplot(1,1,1)
				mpl.set_xticks([])
				mpl.set_yticks([])
				mpl.grid(False)
				mpl.set_xlim(0, map_.w)
				mpl.set_ylim(map_.h, 0)
				mpl.axis('off')

				mpl  = map_.show_mpl(mpl)
				ClusterSize = Counter(lbls)
				printed = []
				for k in range(0,len(X)):

					x,y = map_.to_pixels(X[k][0],X[k][1])
					mpl.plot(x,y,'%s'%visual_patterns[lbls[k]%len(visual_patterns)])
					if lbls[k] not in printed:
						mpl.text(x, y, '%d'%ClusterSize[lbls[k]], fontsize=20,color=visual_patterns[lbls[k]][1])
						printed.append(lbls[k])

				plt.tight_layout()
				pp = PdfPages('Heirarichal[%d]%s.pdf'%(ct,name))
				plt.savefig(pp, format='pdf')
				pp.close()
				ct+=1
				if ct>10:
					break


		return labels
