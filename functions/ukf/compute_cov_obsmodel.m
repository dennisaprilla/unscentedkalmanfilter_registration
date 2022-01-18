function Sigmaest_yacute_k = compute_cov_obsmodel(state_rotation, moving_dataset_cov, fixed_dataset_cov, length_data)
    state_R = eul2rotm(deg2rad(state_rotation'), 'ZYX');
    
    temp1 = repelem({ state_R * moving_dataset_cov * state_R'}, length_data);
    temp2 = repelem({fixed_dataset_cov}, length_data);
    Sigmaest_yacute_k = blkdiag(temp1{:}) + blkdiag(temp2{:});
end

