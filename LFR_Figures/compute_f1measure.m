function [f1] = compute_f1measure(cluster1, cluster2)
f1 = 0;
for i = 1:size(cluster2, 1)
    precison = size(intersect(cluster2(i,:), cluster1),2) /size(cluster1,2);
    recall = size(intersect(cluster2(i,:), cluster1),2) /size(cluster2(i,:),2);
    if precison == 0 && recall == 0
        f1 = max(f1, 0);
    else
        f1 = max(f1, 2*precison*recall / (precison + recall));
    end
end
