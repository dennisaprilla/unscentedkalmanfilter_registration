function Sigma_i = compute_cov_obsmodel(state_rotation, moving_dataset_cov, fixed_dataset_cov, mean_dist, length_data)
    state_R = eul2rotm(deg2rad(state_rotation'), 'ZYX');
    
    temp1 = repelem({ state_R * moving_dataset_cov * state_R'}, length_data);
    temp2 = repelem({fixed_dataset_cov}, length_data);    
    temp3 = repelem({mean_dist*eye(3)}, length_data);
    
    % Sigma_i = Sigma_yacute_k + Sigma_int_k
    % Sigma_i = ( (R * Sigma_u * R') + Sigma_y ) + Sigma_int_k
    Sigma_i = ( blkdiag(temp1{:}) + blkdiag(temp2{:}) ) + blkdiag(temp3{:});
end

