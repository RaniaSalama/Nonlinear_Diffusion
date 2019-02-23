% This code is taken from: https://www.cs.purdue.edu/homes/dgleich/codes/robust-diffusions/
% and modified to run nonlinear diffusion and use FASHION_MNIST dataset
function [errrate, prediction_per_class, variances] = experiment_usps(digits, rad, nlabeled, ntrials, nclass, diffusion_p, t, h, is_draw, sample_size, knn, generate_knn_graph, method) 
%% Use the USPS data for semi-supervised learning
% This experiment replicates the one done in Zhou 2003 on semi-supervised
% learning on a graph.
% This file is modified to run nonlinear diffusions.

%% Load the data
disp('load Data');
load('data\\minst_fashion.mat');
test_patterns = test_patterns(:,1:sample_size);
test_labels = test_labels(:,1:sample_size);
solve_cvx = 0; % = 0 means to use gurobi, = 1 means use (SLOW) cvx

%% digit subset
patterns = [];
labels = [];
for d = digits
    filt = train_labels(d,:)==1;
    labels = [labels d*ones(1,sum(filt))];
    patterns = [patterns train_patterns(:,filt)];
    filt = test_labels(d,:)==1;
    labels = [labels d*ones(1,sum(filt))];
    patterns = [patterns test_patterns(:,filt)];
end
reallabels = labels;
[~,~,labels] = unique(reallabels);

%% construct the graph
%D = pdist2(patterns',patterns','euclidean');

%D = sqrt(sqdist(patterns,patterns));
%K = exp(-D.^2/(2*rad^2));
%Gf = K - diag(diag(K));
%G = Gf;
disp('load KNN Graph');
[~, n] = size(patterns);

% knn distance.
if generate_knn_graph == 1
    T = KDTreeSearcher(patterns');
    [nn,wnn] = knnsearch(T,patterns','K',knn);
    G = sparse(reshape(repmat(1:n, knn, 1), knn * n, 1), reshape(nn', knn * n, 1), exp(-1 * reshape(wnn', knn * n, 1).^2/(2*rad^2)));
    G(1:1+size(G,1):end) = 0;
    save('knn_graph.mat', 'G')
else
    load('knn_graph.mat')
end

% Stop normalization for now.
d = sum(G,2);
G = diag(1./sqrt(d))*G*diag(1./sqrt(d));
%%
nl = numel(nlabeled);
classes = unique(labels);
nclasses = numel(classes);

disp('start the task ...');
assert(numel(labels) == n);
errrate = zeros(nl, ntrials, 1);
prediction_per_class = zeros(ntrials, nclass);
variances = zeros(ntrials, nclasses);
for j=1:nl
    nlabels = nlabeled(j);
    for i=1:ntrials
        fprintf('trial = %d\n', i);
        %%
        % pick the training set, it needs to have one element from each class.
        p = randperm(n);
        train_set = zeros(n,1);
        pl = labels(p);
        [cp,ia] = unique(pl,'first'); %pl(ia) = c
        train_set(p(ia)) = 1;
        test_set = ~train_set;
        test_labels = labels(test_set)';

        p(ia) = [];
        train_set(p(1:(nlabels - nclasses))) = 1;

        Y = full(sparse(find(train_set),labels(logical(train_set)),1,n,nclasses));
        Yt = diag(1./sqrt(d))*Y;      
       
        % Predict using cuts
        Pred = zeros(size(Y));
        sum_stds_per_class = 0;
        for c=1:nclasses
            s = Yt(:,c);
            assert(all(s >= 0));   
            xt = run_nnlinear_diffusion(G, s, diffusion_p, t, h, method);
            Pred(:,c) = xt;
            sum_stds_per_class = sum_stds_per_class + Pred(:,c);
            variances(i, c) = std(xt);
        end
        fprintf('mean variance is = %0.4f\n', sum_stds_per_class/nclasses);
        [~,Pi] = max(Pred,[],2); % find the largest entry in each row
        pred_labels = Pi(test_set)';
        errs = numel(test_set) - sum(pred_labels == test_labels);
        errrate(j,i,1) = errs/numel(test_set);
        for class = 1:nclass
            pred_labels_class = find(pred_labels == class);
            test_labels_class =  find(test_labels == class);
            errs = length(test_labels_class) - length(intersect(pred_labels_class, test_labels_class));
            prediction_per_class(i, class) = errs/length(test_labels_class);
        end
        fprintf('errrates(%2i) = ', nlabels);
        fprintf('%10.5f ', errrate(j,i,:));
        fprintf('\n');
        if is_draw == 1 && i == 1 % If we want to draw, then draw the first trial only
            figure;
            test_patterns = patterns(:,test_set);
            test_pred1 = Pred(test_set,1);
            test_pred2 = Pred(test_set,2);
            draw_diffusion(test_labels, nclass, 2, test_pred1, test_pred2, test_patterns, 'T-shirt/top', 'Trouser', diffusion_p, 1);
            saveas(gcf,strcat('1-2_p=', num2str(diffusion_p),'.eps'),'epsc')
            %saveas(gcf,strcat('1-2_p=', num2str(diffusion_p),'.png'));
            figure;
            test_pred3 = Pred(test_set,3);
            test_pred4 = Pred(test_set,4);
            draw_diffusion(test_labels, nclass, 2, test_pred3, test_pred4, test_patterns, 'Pullover', 'Dress', diffusion_p, 2);
            saveas(gcf,strcat('3-4_p=', num2str(diffusion_p),'.eps'),'epsc')
            %saveas(gcf,strcat('3-4_p=', num2str(diffusion_p),'.png'));
            figure;
            test_pred6 = Pred(test_set,6);
            test_pred9 = Pred(test_set,9);
            draw_diffusion(test_labels, nclass, 2, test_pred6, test_pred9, test_patterns, 'Sandal', 'Bag', diffusion_p, 3);
            %saveas(gcf,strcat('6-9_p=', num2str(diffusion_p),'.png'));
            saveas(gcf,strcat('6-9_p=', num2str(diffusion_p),'.eps'),'epsc')
        end
    end
    for class = 1:nclass
        fprintf('%10.5f %10.5f\n', mean(prediction_per_class(:, class)), std(prediction_per_class(:, class)));
    end
end
for i = 1:1
    fprintf('%10.5f %10.5f', mean(errrate(:,:,i)), std((errrate(:,:,i))));
    fprintf('\n');
end
%%