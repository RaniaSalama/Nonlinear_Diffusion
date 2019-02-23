function draw_diffusion(test_labels, nclass, images_per_class, pred1, pred2, patterns, xlabel_name, ylabel_name, p, iter)
% First group image by class.
max_images_per_class = 10000;
class = [];
for i = 1:nclass
  f = find(test_labels == i)';
  pad = zeros(max_images_per_class - length(f), 1);
  ff = [f; pad];
  class = [class, ff];
end
rng(0);
for i = 1:nclass
  for j = 1:images_per_class
    x = pred1(class(j, i));
    y = pred2(class(j, i));
    hold on;
    if p == 0.5
        imagesc([x x+0.007],[y+0.007 y], reshape(patterns(:,class(j,i)),28,28));
    else 
        if iter == 3
            imagesc([x x+0.0005],[y+0.007 y], reshape(patterns(:,class(j,i)),28,28));
        else
            imagesc([x x+0.0015],[y+0.0015 y], reshape(patterns(:,class(j,i)),28,28));
        end
    end
    colormap('gray');
  end
end
xlabel(xlabel_name)
ylabel(ylabel_name)
