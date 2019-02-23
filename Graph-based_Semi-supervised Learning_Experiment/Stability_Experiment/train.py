import argparse
import numpy as np
import random

from utils import read_parameters, load_data, calculate_normalized_laplacian, calculate_quality_measures
from nonlinear_diffusion import calculate_nonlinear_diffusion, calculate_two_nonlinear_diffusions, self_learning
import networkx as nx


from sklearn.neighbors import KDTree

parser = read_parameters()	
				
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
np.random.RandomState(args.seed)

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data(args.dataset, args.normalize)
idx_train = [i for i, x in enumerate(train_mask) if x]
idx_val = [i for i, x in enumerate(val_mask) if x]
idx_test = [i for i, x in enumerate(test_mask) if x]
n = adj.shape[0]
print("Done loading the data.")

if args.generate_knn == 1:
	dense_features = features
	Tree = KDTree(dense_features)
	dists,idxs = Tree.query(dense_features, k = args.knighbors_number)
	dists = np.exp(-(dists**2/(2*args.rad**2)))
	G = np.zeros((n, n))
	G[np.tile(range(0, n), args.knighbors_number), np.reshape(idxs, n * args.knighbors_number, 1)] = np.reshape(dists, n * args.knighbors_number, 1)
	# Zero diagonal elements.
	for i in range(0,n):
		G[i, i] = 0.0
	# Make G symmtric!
	Gtranspose = np.transpose(G)
	G = G + Gtranspose - np.sqrt(G * Gtranspose)
	print("Done building knn graph")
	dsqrt = np.diag(np.sqrt(G.sum(axis=0)))
	x,y = np.nonzero(dsqrt)
	nnz = x.shape[0]
	for  i in range(0,nnz):
		if dsqrt[x[i], y[i]] != 0:
			dsqrt[x[i], y[i]] = 1.0 / dsqrt[x[i],y[i]];		
	G = np.dot(np.dot(dsqrt, G), dsqrt);
	np.save('G_'+args.dataset+'_k_'+str(args.knighbors_number)+'_rad_'+str(args.rad)+'.npy', G);
else:
	print("Done loading knn graph")
	G = np.load('G_'+args.dataset+'_k_'+str(args.knighbors_number)+'_rad_'+str(args.rad)+'.npy');

[n, nclass] = labels.shape

outf = open("sigma_change_"+args.dataset+"_"+args.function+".txt", "w")


for sigma in np.arange(0.4, 0.7, 0.01):
	if args.use_two_diffusions == 0:
		G = args.w * G + (1.0-args.w) * adj.A
		GG = np.dot(G, G)
		G = G + GG
		L , pinvD = calculate_normalized_laplacian(G)
		y = np.argmax(labels, 1)
		preds =  calculate_nonlinear_diffusion(idx_train, y, args.t, args.h, args.p, L, pinvD, n, nclass)
	else:
		LF, pinvDF = calculate_normalized_laplacian(G)
		Ladj, pinvDadj = calculate_normalized_laplacian(1.0 * adj.A )
		y = np.argmax(labels, 1)
		if args.self_learning == 0:
			preds = calculate_two_nonlinear_diffusions(idx_train , y, args.t, args.h, args.p1, args.p2, LF, Ladj, sigma, pinvDF, pinvDadj, args.w, n, nclass, args.function)
		else:
			preds = self_learning(idx_train , y, args.t, args.h, args.p1, args.p2, LF, Ladj, sigma, pinvDF, pinvDadj, args.w, n, nclass, y, idx_test, args.rank_based, args.function)
	print("Done calculating the diffusions")

	if args.rank_based == 0:
		# Get predicted labels for each data point.
		pred_labels = np.argmax(preds, 0)
		# Report accuracy on test set.
		y = np.argmax(labels[idx_val,:], 1)
		pred_labels_test = pred_labels[idx_test]
		y = np.argmax(labels[idx_test,:], 1)
		accuracy = calculate_quality_measures(y, pred_labels_test)
		outf.write(str(sigma)+","+str(accuracy)+"\n")
	else:
		sorted_pred = np.argsort(-preds)
		sorted_pred_idx = np.zeros((sorted_pred.shape[0], sorted_pred.shape[1]))
		for i in range(0, sorted_pred.shape[0]):
			for j in range(0, sorted_pred.shape[1]):
				sorted_pred_idx[i,sorted_pred[i][j]] = j
		pred_labels = np.argmin(sorted_pred_idx, 0)
		# Report accuracy on test set.
		y =  np.argmax(labels[idx_val,:], 1)
		pred_labels_test = pred_labels[idx_test]
		y = np.argmax(labels[idx_test,:], 1)
		accuracy = calculate_quality_measures(y, pred_labels_test)
		outf.write(str(sigma)+","+str(accuracy)+"\n")
	
outf.close();

