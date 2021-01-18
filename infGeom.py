from __future__ import division # not needed in Python 3
import numpy as np
import time
import h5py
import scipy.io as sio
import scipy.spatial.distance as spdist
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D

def readTorus(filename):
	mat = sio.loadmat(filename)
	keys = mat.keys()
	return mat['x']

def readProbs(filename):
        f = h5py.File(filename)
	p = np.array(f['p'])
	
	if p.ndim == 3:	
		p = np.swapaxes(p, 0, 2)
		p = np.swapaxes(p, 1, 2)
		p = p.reshape(p.shape[0], p.shape[1]*p.shape[2])
        
	return p

def readDistMatrix(filename, N, K):
	f = h5py.File(filename)
	yRow = np.array(f['yRow']).reshape(1, N*K).flatten() - 1
	yCol = np.array(f['yCol']).reshape(1, N*K).flatten() - 1
	yVal = np.array(f['yVal']).reshape(1, N*K).flatten()
	yVal = np.nan_to_num(yVal)

	ySparse = scipy.sparse.coo_matrix((yVal, (yRow, yCol)), shape=(N, N))
	ySym = maximum(ySparse, ySparse.T)
	return ySym

def maximum (A, B):
	BisBigger = A-B
        BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
        return A - A.multiply(BisBigger) + B.multiply(BisBigger)

def probabilityMeasure1D(embWindow, observable, xi_point):
	N = observable.shape[0]
	n = N - embWindow + 1
	p = np.zeros( (n, len(xi_point)))

#	grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth':np.linspace(0.2,0.2,1)}, cv=2)

	for i in range(embWindow, N + 1):
		print i
		Xi = observable[i - embWindow : i].reshape(-1, 1)
		kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(Xi)
#		grid.fit(Xi)
#		print grid.best_params_
#		kde = grid.best_estimator_
		log_pdf = kde.score_samples(xi_point)
		pdf = np.exp(log_pdf)
		p[i - embWindow, :] = pdf/np.sum(pdf)

	return p

def plotProbabilityMeasures(xi_point, probs):
	plt.figure()
	for i in range(1,10):
		plt.plot(xi_point, probs[i, :])
	plt.show(block=False)

def calcHellingerDistance(probs, K):
	N = probs.shape[0]
	probs = np.sqrt(probs)
	print 'Euclidean distance ...'
#	D = spdist.squareform( spdist.pdist(probs, 'euclidean')) # slower than euclidean_distances
	D = euclidean_distances(probs)
	D = 1/np.sqrt(2)*D

	# check the matrix is symmetric
	print sum(sum(D - D.T))

	# sorting distances
	print 'Sorting indices ...'
	idx = np.argsort(D)

	rows = np.matlib.repmat(np.arange(0,N),K,1).T
	yInd = idx[:, 0 : K]
	yVal = D[rows, yInd]
	
#	yRow = np.matlib.repmat(np.arange(0,N).reshape(-1,1),K, 1).reshape(1, N*K).flatten()
	yRow = rows.reshape(1, N*K).flatten()
	yCol = yInd.reshape(1, N*K).flatten()
	yVal = yVal.reshape(1, N*K).flatten()

	ySparse = scipy.sparse.coo_matrix((yVal, (yRow, yCol)), shape=(N, N))
	
	print 'Symmetrize distance matrix ...'
	ySym = maximum(ySparse, ySparse.T) # solution to np.maximum for sparse matrices
	
#	print 'Saving distance matrix ...'
#	sio.savemat('ySym.mat', mdict={'ySym':ySym})

	return ySym

def eigenfunctionsProb(ySym, alpha, nbEigs, epsilon):
	print 'Exponential ...'	
	[yRow, yCol, yVal] = scipy.sparse.find(ySym)
	yVal = yVal**2
	yVal = np.exp( -yVal / epsilon**2 )	

	N = ySym.shape[0]
	l = scipy.sparse.coo_matrix((yVal, (yRow, yCol)), shape=(N, N))	

	print 'Alpha normalization ...'
	if alpha != 0:
		if alpha != 1:
			q = np.sum(l, 0)**alpha	
		else:	
			q = np.sum(l, 0)

		q = np.asarray(q).flatten()
		yVal = yVal / q[yRow] / q[yCol]
		l = scipy.sparse.coo_matrix((yVal, (yRow, yCol)), shape=(N, N))

	print 'Laplacian normalization ...'
	d = np.sum(l, 0)
	d = np.asarray(d).flatten()
	if alpha == 0:
		q = d
	rootD = np.sqrt(d)
	yVal = yVal / rootD[yRow] / rootD[yCol]
	l = scipy.sparse.coo_matrix((yVal, (yRow, yCol)), shape=(N, N)) # always symmetric?

	print 'Eigenfunctions ...'
	[eigvals, v] = scipy.sparse.linalg.eigs(l, nbEigs) # already sorted in descending order
						# returned eigenvalues and eigenvectors are complex
						# use only the real part, the imaginary part is zero
	v = v.real
	eigvals = eigvals.real
	mu = v[:, 0]**2
	v = v/np.matlib.repmat(np.matrix(v[:,0]).T, 1, nbEigs)
	sio.savemat('eigs.mat', mdict={'v':v})
	return v	

def plotEigenfunctions(v):
	for i in range(0,5):
		plt.figure()
		plt.plot(v[:,i])
		plt.show(block=False)

def plotTorusEigenfunctions(x, v, embWindow):
	fig, ax = plt.subplots()
	plt.get_current_fig_manager().window.setGeometry(0,0,1400,1200)

	for i in range(1,17):
		ax = fig.add_subplot(4,4,i, projection='3d')

#		ax.scatter(x, y, z, c='r', marker='o')
#		scatter3(x(1,embWindow:end),x(2,embWindow:end),x(3,embWindow:end),5,v(:,i));
		x_emb = x[:, embWindow - 1 : x.shape[1]]
        	idx = np.argsort(x_emb[2,:])
		colors = np.asarray(v[idx,i]).flatten()
		# plot in 3D and then rotate the axis
        	sc = ax.scatter( x_emb[0, idx], x_emb[1, idx],
				x_emb[2, idx], s = 5, c = colors, 
				edgecolor = 'black', linewidth = '0', cmap = 'jet');

#		ax.view_init(0, 90)
		ax.azim = -90
		ax.elev = 90
		plt.xlim([-1.5, 1.5])
		plt.ylim([-1.5, 1.5])
#		ax.w_zaxis.line.set_lw(0.) # remove only z axis
#		ax.set_zticks([])
		ax.axis("off")
		plt.colorbar(sc)

	plt.show(block=False)	

def main():
        embWindow = 80
        binSize = 1/50
	minS, maxS = -1.5, 1.5
	K = 5000
	alpha = 1
	nbEigs = 20
	epsilon = 0.5

	filenameData = 'torus128000_freq5.4772.mat'
	x = readTorus(filenameData)
#	observable = x[0,:]
#	xi_point_samples = (maxS - minS)/binSize + 1
#	xi_point = np.linspace(minS, maxS, xi_point_samples).reshape(-1, 1)
	
#	probs = probabilityMeasure1D(embWindow, observable, xi_point)
#	sio.savemat('probsTest.mat', mdict={'probs':probs, 'xi_point':xi_point})
	filenameProbs = './../../torusProbMeasure_x1x2_freq5.4772_1000_128_emb80_probBins61_-inf_-inf_inf_inf.mat'
	probs = readProbs(filenameProbs) # choose KDE bandwidth
#	plotProbabilityMeasures(xi_point, probs)
	
	start_time = time.time()
	ySym = calcHellingerDistance(probs, K)
	exec_time = (time.time() - start_time)/60
        print 'Distance computation time: ' + str(exec_time)
#	filenameDist = './../../torusDistProb_x1x2_freq5.4772_1000_128_emb80_K1000_bins50_-1.5_1.5.mat'
#	N = 127925
#	ySym = readDistMatrix(filenameDist, N, K)

	start_time = time.time()
	v = eigenfunctionsProb(ySym, alpha, nbEigs, epsilon)
	exec_time = (time.time() - start_time)/60
	print 'Eigenfunctions computation time: ' + str(exec_time)
#	plotEigenfunctions(v)
	plotTorusEigenfunctions(x, v, embWindow)
