function [test_set,test_labels,train_set,train_labels] = build_test_train(labels,nlabels,nerrors)
% BUILD_TEST_TRAIN Create a testing set and training set from a set of
% labels.
%
% [test_set,test_labels,train_set,train_labels] 
%   = build_test_train(labels,nlabels,nerrors)
% where 
%   labels is a vector with integer labels for each class
%   nlabels is the desired number of labels to use total. There will be at
%     least one from each class. 
%   nerrors is the number of errors to make in the labels, if this value is
%    between 0 and 1 (excluding the endpoints), then it's a fraction of the
%    total labels, but ensures that there is at least one correct label in
%    each class.
%
% test_set is the set of items to evaluate for performance
% test_labels are the true labels in the test_set
% train_set are the training set of items
% train_labels are the labels of the training items, including any errors
%   (you can see the errors with labels(train_set) ~= train_labels
% 

if nargin<3
    nerrors = 0;
end

n = numel(labels);
p = randperm(n);
train_set = false(n,1);
pl = labels(p);
[cp,ia] = unique(pl,'first'); %pl(ia) = c
nclasses = numel(cp);

train_set(p(ia)) = 1;
p(ia) = [];
train_set(p(1:(nlabels - nclasses))) = 1;

train_labels = labels(train_set);
if nerrors > 0
    if nerrors < 1
        nerrors = ceil(nerrors*(nlabels - nclasses));
    end
    assert(nlabels >= nclasses)
    assert(nerrors <= nlabels - nclasses);
    validerrors = p(1:(nlabels - nclasses));
    % introduce nerrors into the training set in 
    rp = randperm(numel(validerrors));
    erroridx = validerrors(rp(1:nerrors)); 
    labels_err = labels;
    labels_err(erroridx) = cp(randi(nclasses,nerrors,1));
    train_labels = labels_err(train_set);
end

test_set = ~train_set;
test_labels = labels(test_set)';

assert(sum(test_set.*train_set) == 0);
assert(sum(train_set) == nlabels);
