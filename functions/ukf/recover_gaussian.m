function [mu, sigma] = recover_gaussian(sigma_points, w_m, w_c)
% This function computes the recovered Gaussian distribution (mu and sigma)
% given the sigma points (size: nx2n+1) and their weights w_m and w_c:
% w_m = [w_m_0, ..., w_m_2n], w_c = [w_c_0, ..., w_c_2n].
% The weight vectors are each 1x2n+1 in size,
% where n is the dimensionality of the distribution.
% This function can be used too for either transformed sigma points by f()
% or by h()

% Try to vectorize your operations as much as possible

% TODO: compute mu
mu = sum(w_m .* sigma_points, 2);

% TODO: compute sigma
sigma = zeros(length(mu));

% there is another way to do this in matlab, check comment below
for j=1:size(sigma_points,2)
    temp = w_c(j) * (sigma_points(:,j) - mu) * (sigma_points(:,j) - mu)';
    sigma = sigma+temp;
end


end

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  