clear;
clc;
% Create train and test dataset
 images = loadMNISTImages('data\\fashion-mnist\\train-images-idx3-ubyte');
 labels = loadMNISTLabels('data\\fashion-mnist\\train-labels-idx1-ubyte');
 images = [images, loadMNISTImages('data\\fashion-mnist\\t10k-images-idx3-ubyte')];
 labels = [labels; loadMNISTLabels('data\\fashion-mnist\\t10k-labels-idx1-ubyte')];
 nlabels = 10;
 nerrors = 0;
 [test_set,test_labels,train_set,train_labels] = build_test_train(labels,nlabels,nerrors);
 train_patterns = images(:,train_set);
 test_patterns = images(:,test_set);
 train_no = size(train_patterns);
 train_labels_no = train_no(2);
 train_labels_matrix = zeros(10, train_labels_no);
 for i = 1:train_labels_no
   train_labels_matrix(train_labels(i)+1, i) = 1;
 end
 train_labels = train_labels_matrix;
 test_no = size(test_patterns);
 test_labels_no = test_no(2);
 test_labels_matrix = zeros(10, test_labels_no);
 for i = 1:test_labels_no
   test_labels_matrix(test_labels(i)+1, i) = 1;
 end    
 test_labels = test_labels_matrix;
 save('data\\minst_fashion.mat', 'train_patterns', 'train_labels', 'test_patterns', 'test_labels');

% Run nonlinear diffusion with p = 0.5.
rng('default');
sigma = 1.25;
nlabeled = 10;
ntrials = 1;
classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
nclass = 10;
p = 0.5;
t = 5000;
h = 0.001;
is_draw = 1;
sample_size = 1000;
knn = 100;
generate_knn_graph = 0;
method = 0;
[errrate_nnlinear, prediction_per_class_nnlinear, variance_nnlinear] = experiment_usps(classes, sigma, nlabeled, ntrials, nclass, p, t, h, is_draw, sample_size, knn,generate_knn_graph, method);
% Run linear diffusion p = 1.0.
rng('default');
p = 1.0;
[errrate_linear, prediction_per_class_linear, variance_linear] = experiment_usps(classes, sigma, nlabeled, ntrials, nclass, p, t, h, is_draw, sample_size, knn,generate_knn_graph, method);