function [cond, min_indx, cluster] = compute_conductance(A, f)
% Taken from Austin code
[~, order] = sort(f);
B = A(order, order);
B_lower = tril(B);
B_sums = full(sum(B, 2));
B_lower_sums = sum(B_lower, 2);
volumns = cumsum(B_sums);
num_cuts = cumsum(B_sums - 2 * B_lower_sums);
total_vols = full(sum(sum(A)));
volumns_other = total_vols * ones(length(order), 1) - volumns;
vols = min(volumns, volumns_other);
scores = num_cuts ./ vols;
[cond, min_indx] = min(scores);

n = size(A, 1);
if min_indx < floor(n/2)
    cluster = order(1:min_indx)';
else 
    cluster = order((min_indx+1):n)';
end
end

