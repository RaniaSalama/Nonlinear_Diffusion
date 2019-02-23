function f = run_nnlinear_diffusion(A, u, p, t, h, method)
  % method = 0 equals power function, method = 1 equals tanh function
  D = diag(sum(A));
  n = size(A,1);
  [x, y] = find(D);
  pinvD = D;
  for i = 1:length(x)
    pinvD(x(i), y(i)) = 1.0/pinvD(x(i), y(i));
  end
  L = speye([n n]) -  A * pinvD;
  for tt = 1:t
     if method == 0
        u = u - h * L * (u.^p) ;
        % To ensure u is between 0 and 1 at any time.
        u(u < 0.0) = 0.0;
        u(u > 1.0) = 1.0; 
     elseif method == 1
        u = u - h * L * tanh(u) ;
     end
  end
  f = pinvD * u.^p;
