using Knet, NearestNeighbors, JLD
include(Knet.dir("data","fashion-mnist.jl"))

# Parameters.
nclass = 10
trial_number = 10
rad = 1.25
h = 0.001
dim = 28 * 28
sample_size = 1000;
knighbors_number = 100
p_array = [0.5, 1]
t = 1000
labeled_examples_per_class = 10
generate_knn_graph = 1
for p = p_array
	file_output = open(string("p=", p, "_variances_100.txt"), "w");
	srand(0)
	# Retrieve Data, features by samples
	xtrain, ytrain, xtest, ytest, labels = fmnist()
	xtrain = reshape(xtrain, dim, size(xtrain, 4))
	xtest = reshape(xtest, dim, size(xtest, 4))
	println("Done retrieving the data")
	# Add test and train together.
	x_all = [xtrain xtest]
	y_all = [ytrain;ytest]
	x_all = x_all[:, 1:sample_size]
	y_all = y_all[1:sample_size]
	n = size(x_all, 2)
	println("Done adding testing and training together")
	# Construct KNN graph.
	if generate_knn_graph == 1
		Tree = KDTree(x_all)
		idxs, dists = knn(Tree, x_all, knighbors_number, true)
		G = sparse(collect(Iterators.flatten(repmat(1:n, 1, knighbors_number)')), collect(Iterators.flatten(idxs)), exp.(-collect(Iterators.flatten(dists)).^2/(2*rad^2)));
		# Zero diagonal elements.
		for i = 1:n
			G[i, i] = 0.
		end
		# Make G symmtric!
		G = (G + G' - sqrt.(G.*G'))
		save(string("knn_graph_k=", knighbors_number, "_fixed.jld"),"G",G)
		println("Done building knn graph")
		generate_knn_graph = 0
	else
		dict = load(string("knn_graph_k=", knighbors_number, "_fixed.jld"))
		G = dict["G"]
		range = collect(1:n)
		dsqrt = sparse(range, range, sqrt.(vec(sum(G,2))))
		x,y = findn(dsqrt)
		for i = 1:length(x)
			if dsqrt[x[i], y[i]] != 0
				dsqrt[x[i], y[i]] = 1.0 / dsqrt[x[i],y[i]];		
			end
		end
		G = dsqrt * G * dsqrt;
		println("Done loading knn graph")
	end
	range = collect(1:n)
	pinvD = sparse(range, range, vec(sum(G,2)))
	pinvD = convert(SparseMatrixCSC{Float64}, pinvD)
	x, y = findn(pinvD)
	for i = 1:length(x)
		if pinvD[x[i], y[i]] != 0
			pinvD[x[i], y[i]] = 1.0/pinvD[x[i], y[i]];
		end
	end
	L = speye(n,n) - G * pinvD
	println("Done calculating the laplacian")
	error_rates = zeros(trial_number, 1)
	prediction_per_class = zeros(trial_number, nclass)
	variances = zeros(trial_number, nclass)
	for i = 1:trial_number
		@printf("Trial = %d\n", i)
		perm = randperm(n)		
		# Get train set.
		train_set_indx = zeros(Int, labeled_examples_per_class * nclass, 1)
		index = 1
		class_count = zeros(Int, n, 1);
		for j = 1:n
			if class_count[y_all[perm[j]]] < labeled_examples_per_class
				# Add to the train.
				train_set_indx[index] = perm[j]
				index = index + 1
				class_count[y_all[perm[j]]] = class_count[y_all[perm[j]]] + 1
			end
			if index > labeled_examples_per_class * nclass
				break
			end
		end
		test_set_indx = setdiff(1:n, train_set_indx)
		
		preds = zeros(Float64, nclass, n)
		# Start diffusion
		for j = 1:nclass*labeled_examples_per_class
			@printf("Start from class = %d\n", j)
			u = sparsevec([train_set_indx[j]], [1], n, +)
			for tt = 1:t
				u = u - h * L * (u .^ p)
				u[u .< 0] = 0.
				u[u .> 1] = 1.			 
			end
			f = pinvD * u
			preds[y_all[train_set_indx[j]],:] = max.(preds[y_all[train_set_indx[j]],:], f)
		end
		# Get predicted labels for each data point.
		vals, inds = findmax(preds, 1)
		pred_labels = map(x->ind2sub(preds, x)[1], inds)
		pred_labels = pred_labels[test_set_indx]
		y_test = y_all[test_set_indx]
		error_rate = (length(y_test) - sum(pred_labels .==  y_test))/length(y_test);
		error_rates[i] = error_rate
		for c = 1:nclass
			pred_labels_class = find(pred_labels .== c);
			test_labels_class =  find(y_test .== c);
			errs = length(test_labels_class) - length(intersect(pred_labels_class, test_labels_class));
			prediction_per_class[i, c] = errs/length(test_labels_class);
			variances[i, c] = std(preds[c,:])
		end
	end
	for i = 1:trial_number
		for cc = 1:nclass
			print(file_output, string(variances[i,cc], " "))
		end
		print(file_output, "\n")
	end				
	print(file_output, "\n")
	flush(file_output)
end
