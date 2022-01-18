function ytilde_k = h_transform(sigma_points, u_i)
% This function applies a transformation to a set of sigma points.

ytilde_k = [];

for j=1:size(sigma_points,2)
    xest_bar_trans = sigma_points(1:2, j);
    xest_bar_theta = sigma_points(3, j);
    
    R_xest_bar_theta = [ cosd(xest_bar_theta) -sind(xest_bar_theta); ...
                         sind(xest_bar_theta)  cosd(xest_bar_theta) ];
    temp = R_xest_bar_theta * u_i + xest_bar_trans;
    
    % I need to make 3d points become 1d vector, as described in eq. 26
    ytilde_k = [ytilde_k, reshape(temp, [], 1)];
end


end

