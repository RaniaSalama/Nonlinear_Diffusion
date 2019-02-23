function [u] = run_diffusion(s , L, h, t, p, method, D)
    n = size(L, 2);
    u = sparse(zeros(n, 1));
    u(s) = 1; 
    for tt = 1:t
      if strcmp(method, 'power') == 1
          u = u - h * L * (u.^p) ;
          u(u < 0.0) = 0.0;
          u(u > 1.0) = 1.0; 
      elseif strcmp(method, 'tanh') == 1
          u = u - h * L * u;
          u = tanh(u);
      elseif strcmp(method, 'plaplacian') == 1
          diff_u = L * D^(-1) * u;
          u = u - h * L' * ((abs(diff_u)).^(p-1) .* sign(diff_u)) ;
          u(u < 0.0) = 0.0;
          u(u > 1.0) = 1.0;
      end
    end
    u = D^(-1) * u;
end



