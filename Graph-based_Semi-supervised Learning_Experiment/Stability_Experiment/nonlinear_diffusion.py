import numpy as np

def calculate_nonlinear_diffusion(idx_train, labels, t, h, p, L, pinvD, n, nclass, nnlinear_function):
	preds = np.zeros((nclass, n))
	# Start diffusion
	labels_train = labels[idx_train]
	for j in range(0, nclass):
		indexes = [idx_train[i] for i,x in enumerate(labels_train) if x == j]
		u = np.zeros(n)
		u[indexes] = 1.0 / (len(indexes))
		for tt in range(0, t):
			if nnlinear_function == "power":
				u = u - h * np.dot(L, np.power(u, p))
				u[u < 0] = 0.0
				u[u > 1] = 1.0	
			if nnlinear_function == "tanh":
				u = u - h * np.dot(L, np.tanh(u))
		f = u
		train_j_class = j
		preds[train_j_class,:] = np.maximum(preds[train_j_class,:], f)
	return preds
	
def calculate_two_nonlinear_diffusions(idx_train, labels, t, h, p1, p2, LF, Ladj, sigma, pinvDF, pinvDadj, w, n, nclass, nnlinear_function):
	preds = np.zeros((nclass, n))
	labels_train = labels[idx_train]
	# Start diffusion
	for j in range(0, nclass):
		indexes = [idx_train[i] for i,x in enumerate(labels_train) if x == j]
		uF = np.zeros(n)
		uF[indexes] = 1.0 / (len(indexes))
		uadj = np.zeros(n)
		uadj[indexes] = 1.0 / (len(indexes))
		print("Running nonlinear diffusion from class = %d"% (j))
		for tt in range(0, t):
			if nnlinear_function == "power":
				uadj = uadj - h * np.dot(Ladj, np.power(uadj, p1)) - sigma * (uadj - uF)
				uF = uF - h * np.dot(LF, np.power(uF, p2)) - sigma * (uF - uadj)
				uF[uF < 0] = 0.0
				uF[uF > 1] = 1.0
				uadj[uadj < 0] = 0.0
				uadj[uadj > 1] = 1.0
			if nnlinear_function == "tanh":
				uadj = uadj - h * np.dot(Ladj, uadj) - sigma * (uadj - uF)
				uadj = np.tanh(uadj)
				uF = uF - h * np.dot(LF, uF) - sigma * (uF - uadj)
				uF = np.tanh(uF)
		# uadj and uf should be close ....		
		train_j_class = j
		f = w * np.dot(pinvDF, uF) + (1-w) * np.dot(pinvDadj, uadj)
		preds[train_j_class,:] = np.maximum(preds[train_j_class,:], f)
	return preds

def choose_samples(preds, samples_no, remove_colums):
	# Choose samples such that we preserve class distribution.
	[nclass, n] = preds.shape
	preds[:, remove_colums] = -1
	sorted_index = np.argsort(preds)
	selected_samples = np.array([])
	for i in range(0, nclass):
		samples_per_class = int(samples_no / nclass);
		selected_samples_i = np.squeeze(sorted_index[i,n-samples_per_class:n])
		selected_samples = np.unique(np.append(selected_samples, selected_samples_i)).astype(int)
	return selected_samples

def self_learning(idx_train , y, t, h, p1, p2, LF, Ladj, sigma, pinvDF, pinvDadj, w, n, nclass, labels, idx_test, rank_based, nnlinear_function):
	preds = calculate_two_nonlinear_diffusions(idx_train, y, t, h, p1, p2, LF, Ladj, sigma, pinvDF, pinvDadj, w, n, nclass, nnlinear_function)
	if rank_based == 0:
		pred_labels = np.argmax(preds, 0)
	else:
		sorted_pred = np.argsort(-preds)
		sorted_pred_idx = np.zeros((sorted_pred.shape[0], sorted_pred.shape[1]))
		for i in range(0, sorted_pred.shape[0]):
			for j in range(0, sorted_pred.shape[1]):
				sorted_pred_idx[i,sorted_pred[i][j]] = j
		pred_labels = np.argmin(sorted_pred_idx, 0)
	new_samples = idx_train;
	for iter in range(5):
		selected_samples = choose_samples(preds, 10, new_samples)
		pred_labels[idx_train] = labels[idx_train]
		new_samples = np.unique(np.concatenate((selected_samples, new_samples)));
		preds =  calculate_two_nonlinear_diffusions(new_samples, pred_labels, t, h, p1, p2, LF, Ladj, sigma, pinvDF, pinvDadj, w, n, nclass, nnlinear_function)
		if rank_based == 0:
			pred_labels = np.argmax(preds, 0)
		else:
			sorted_pred = np.argsort(-preds)
			sorted_pred_idx = np.zeros((sorted_pred.shape[0], sorted_pred.shape[1]))
			for i in range(0, sorted_pred.shape[0]):
				for j in range(0, sorted_pred.shape[1]):
					sorted_pred_idx[i,sorted_pred[i][j]] = j
			pred_labels = np.argmin(sorted_pred_idx, 0)	
	return preds
