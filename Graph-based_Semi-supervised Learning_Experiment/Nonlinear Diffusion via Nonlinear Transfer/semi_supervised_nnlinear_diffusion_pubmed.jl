using NearestNeighbors, JLD, NPZ
include("util.jl")
include("self_learning.jl")


# Parameters.
nclass = 3
dataset = "pubmed"
rad = 1.0
h = 0.001
knighbors_number = 100
t = 400
normalize = 0
generate_knn_graph = 1
sigma = 0.04
p1 = 1.9
p2 = 1.7
w = 0.5
use_weights = 0
srand(0)

# Retrive Data.
adj = npzread("data/pubmed/adj.npz")
features = readdlm("data/pubmed/features.txt")
y_train = readdlm("data/pubmed/y_train.txt")
y_val = readdlm("data/pubmed/y_val.txt")
y_test = readdlm("data/pubmed/y_test.txt")
train_mask = readdlm("data/pubmed/train_mask.txt")
val_mask = readdlm("data/pubmed/val_mask.txt")
test_mask = readdlm("data/pubmed/test_mask.txt")
labels = readdlm("data/pubmed/labels.txt")
n = size(features, 1)
# Construct KNN graph.
if generate_knn_graph == 1
	Tree = KDTree(transpose(features))
	idxs, dists = knn(Tree, transpose(features), knighbors_number, true)
	G = sparse(collect(Iterators.flatten(repmat(1:n, 1, knighbors_number)')), collect(Iterators.flatten(idxs)), exp.(-collect(Iterators.flatten(dists)).^2/(2*rad^2)));
	# Zero diagonal elements.
	for i = 1:n
		G[i, i] = 0.
	end
	# Make G symmtric!
	G = (G + G' - sqrt.(G.*G'))
	range = collect(1:n)
	dsqrt = inverseDegreeMatrix(sparse(range, range, sqrt.(vec(sum(G,2)))))
	G = dsqrt * G * dsqrt;
	save(string("knn_graph_k=", knighbors_number,"_", dataset, "_fixed.jld"), "G", G)
	println("Done building knn graph")
	generate_knn_graph = 0
else
	dict = load(string("knn_graph_k=", knighbors_number, "_", dataset, "_fixed.jld"))
	G = dict["G"]
	println("Done loading knn graph")
end
range = collect(1:n)
pinvDG = inverseDegreeMatrix(sparse(range, range, vec(sum(G,2))))
pinvDadj = inverseDegreeMatrix(sparse(range, range, vec(np.sum(adj,1) * 1.0)))
node1, node2 = findnz(triu(G))
m = length(node1)
em = ones(m)
if use_weights == 1
	for i = 1:m
		em[i] = G[node1[i], node2[i]]
	end
end
edges_idx = collect(1:m)
NG = sparse(vcat(edges_idx,edges_idx),[node1; node2], vcat(em, -1 * em))
node1, node2 = np.nonzero(adj)
node1 = node1 + 1
node2 = node2 + 1
m = length(node1)
em = 1.0*ones(m)
edges_idx = collect(1:m)
Nadj = sparse(vcat(edges_idx,edges_idx),[node1; node2], vcat(em, -1 * em))
println("Done calculating the laplacians.")

max_val, max_idx = findmax(y_train, 2)
train_labels = ind2sub(size(y_train), vec(max_idx))[2]
idx_train = range[train_mask]
max_val, max_idx = findmax(y_test, 2)
y_test = ind2sub(size(y_test), vec(max_idx))[2]
y_test = y_test[test_mask]
println("Running nonlinear diffusion using p-laplacians.")
preds = run_nonlinear_diffusion(pinvDG, NG, pinvDadj, Nadj, n, nclass, sigma, w, idx_train, train_labels, p1, p2, h)
# Get predicted labels for each data point.
vals, inds = findmax(preds, 1)
pred_labels = map(x->ind2sub(preds, x)[1], inds)
pred_labels = pred_labels[test_mask]
accuracy = sum(pred_labels .==  y_test)/length(y_test)
@printf("accuracy = %6.4f\n", accuracy)

println("Running nonlinear diffusion using p-laplacians with self_learning")
preds = self_learning(pinvDG, NG, pinvDadj, Nadj, n, nclass, sigma, w, idx_train, train_labels, p1, p2, h, test_mask, y_test)
# Get predicted labels for each data point.
vals, inds = findmax(preds, 1)
pred_labels = map(x->ind2sub(preds, x)[1], inds)
pred_labels = pred_labels[test_mask]
accuracy = sum(pred_labels .==  y_test)/length(y_test)
@printf("accuracy = %6.4f\n", accuracy)

