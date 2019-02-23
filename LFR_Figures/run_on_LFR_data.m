rng('default');
n = 1000;
t = 100;
d = 10;
h = 0.001;
trial_num = 100;
conds_mean = zeros(5, 5);
conds_std = zeros(5, 5);
f1_mean = zeros(5, 5);
f1_std = zeros(5, 5);

index = 1;
for mu = 0.02:0.02:0.5
    mu
    graph_file = fopen(strcat('LFR_data_1000_more_more_mu\mu=', num2str(mu), '\network.dat'));
    community_file = fopen(strcat('LFR_data_1000_more_more_mu\mu=', num2str(mu), '\community.dat'));
    edges = textscan(graph_file,'%d\t%d');
    fclose(graph_file);
    m = size(double(edges{1}));
    A = sparse(double(edges{1}), double(edges{2}), ones(m(1), 1));
    D = sparse(diag(sum(A)));
    P =  sparse(A * D^(-1));
    L = (speye([n n]) -  P);
    [node1, node2] = find(triu(A));
    m = length(node1);
    em = ones(1, m);
    edges_idx = 1:m;
    N = sparse([edges_idx,edges_idx]',[node1; node2], [em, -1 * em]', m, n);

    ground_truth = ones(n, n);
    node_community = zeros(n, d);
    C = 0;
    for i = 1:n
       line = fgetl(community_file);
       split_line = strsplit(line, '\t');
       split1 = split_line(1);
       node = str2double(split1{1});
       split2 = split_line(2);
       communities = str2double(strsplit(strtrim(split2{1}), ' '));
       C = max(C, communities);
       ground_truth(communities, ground_truth(communities,1)+1) = node;
       ground_truth(communities,1) = ground_truth(communities,1) + 1;
       node_community(i,1:size(communities,2)) = communities;
    end
    fclose(community_file);

    conds = zeros(trial_num, 5);
    f1 = zeros(trial_num, 5);
    % Start diffusion from each node and measure conuctance and F1.
    for i = 1:trial_num
         s = randi(n);
         s_cluster_id_all = node_community(s,:);
         s_cluster_id = [];
         for q = 1:size(s_cluster_id_all, 2)
             if s_cluster_id_all(q) ~= 0
                s_cluster_id = [s_cluster_id, s_cluster_id_all(q)];
             end
         end
         groun_truth_cluster = ground_truth(s_cluster_id, 2:ground_truth(s_cluster_id, 1));
         u_linear =  run_diffusion(s , L, h, t, 1, 'power', D);
         [cond, ~, cluster] = compute_conductance(A, u_linear);
         conds(i, 1) = cond;
         f1(i, 1) = compute_f1measure(cluster, groun_truth_cluster);
         u_nnlinear_5 = run_diffusion(s , L, h, t, 0.5, 'power', D);
         [cond, ~, cluster] = compute_conductance(A, u_nnlinear_5);
         conds(i, 2) = cond;
         f1(i, 2) = compute_f1measure(cluster, groun_truth_cluster);      
         u_nnlinear_tanh =  run_diffusion(s , L, h, t, 2.0, 'tanh', D);
         [cond, ~, cluster] = compute_conductance(A, u_nnlinear_tanh);
         conds(i, 3) = cond;
         f1(i, 3) = compute_f1measure(cluster, groun_truth_cluster);
         u_p_laplacian =  run_diffusion(s , N, h, t, 1.9, 'plaplacian', D);
         [cond, ~, cluster] = compute_conductance(A, u_p_laplacian);
         conds(i, 4) = cond;
         f1(i, 4) = compute_f1measure(cluster, groun_truth_cluster);
         % personalized pagerank
         u = zeros(n, 1);
         u(s) = 1;
         p = pagerank(A,struct('alg','approx', 'x0', u, 'v', u));
         [cond, ~, cluster] = compute_conductance(A, p);
         conds(i, 5) = cond;
         f1(i, 5) = compute_f1measure(cluster, groun_truth_cluster);
    end
    conds_mean(index,:) = mean(conds, 1);
    conds_std(index,:) = std(conds, 1);
    f1_mean(index,:) = mean(f1, 1);
    f1_std(index,:) = std(f1, 1);
    index = index + 1;
end
% Draw conductance figure.
fig = figure;
colors = ['b', 'r', 'y', 'g', 'c', 'g'];
markers = ['o','x','+', '*', 'd', 's'];
for i=5:-1:1
    e = shadedErrorBar(0:0.02:0.5, [0; conds_mean(:,i)], [0; conds_std(:,i).^2], 'lineProps', colors(i));
    e.Marker = markers(i);
    e.Color = colors(i);
    hold on;   
end
hold on;
h1 = plot(NaN, NaN, 'b');
hold on;
h2 = plot(NaN, NaN, 'r');
hold on;
h3 = plot(NaN, NaN, 'y');
hold on;
h4 = plot(NaN, NaN, 'g');
hold on;
h5 = plot(NaN, NaN, 'c');
hold on;
l = legend([h1, h2, h3, h4, h5],{'hk', 'p=0.5', 'tanh', 'plaplacian', 'ppr'});
set(l, 'FontSize', 14);
legend('Location', 'northwest');
xlabel('\mu', 'FontSize', 18)
ylabel('Conductance', 'FontSize', 18)
set(gca, 'XTick', 0:0.1:0.5, 'fontsize', 18)
ylim([0, 1])
legend boxoff
set(fig,'Units','Inches');
pos = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig,'conductance_lfr_more_mu','-dpdf','-r0')

% Draw F1 measure figure.
fig = figure;
colors = ['b', 'r', 'y', 'g', 'c', 'g'];
markers = ['o','x','+', '*', 'd', 's'];
for i=5:-1:1
    e = shadedErrorBar(0:0.02:0.5, [0; f1_mean(:,i)], [0; f1_std(:,i).^2], 'lineProps', colors(i));
    e.Marker = markers(i);
    e.Color = colors(i);
    hold on; 
end
hold on;
h1 = plot(NaN, NaN, 'b');
hold on;
h2 = plot(NaN, NaN, 'r');
hold on;
h3 = plot(NaN, NaN, 'y');
hold on;
h4 = plot(NaN, NaN, 'g');
hold on;
h5 = plot(NaN, NaN, 'c');
hold on;
l = legend([h1, h2, h3, h4, h5], {'hk', 'p=0.5', 'tanh', 'plaplacian', 'ppr'});
set(l, 'FontSize', 14);
legend('Location', 'northeast');
xlabel('\mu', 'FontSize', 18)
ylabel('F1 measure', 'FontSize', 18)
set(gca, 'XTick', 0:0.1:0.5, 'fontsize', 18)
ylim([0, 1])
legend boxoff
set(fig,'Units','Inches');
pos = get(fig,'Position');
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(fig,'f1_lfr_more_mu','-dpdf','-r0')
