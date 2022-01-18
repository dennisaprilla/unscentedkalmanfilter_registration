function Sigma_x_k = compute_cov_statemodel(fixed_dataset, obsmodel_covariance)

% noise of process model (Eq. 27)
c = sqrt(180);
lambda_for_SigmaX = var(fixed_dataset, [], 2);
var_for_SigmaX    = diag(obsmodel_covariance);

Sigma_x_k  = diag( [ var_for_SigmaX(1), var_for_SigmaX(2), var_for_SigmaX(3), ...
                     c*c / ( sqrt(lambda_for_SigmaX(2)/var_for_SigmaX(3)) + sqrt(lambda_for_SigmaX(3)/var_for_SigmaX(2)) ) , ...
                     c*c / ( sqrt(lambda_for_SigmaX(1)/var_for_SigmaX(3)) + sqrt(lambda_for_SigmaX(3)/var_for_SigmaX(1)) ) , ...
                     c*c / ( sqrt(lambda_for_SigmaX(1)/var_for_SigmaX(2)) + sqrt(lambda_for_SigmaX(2)/var_for_SigmaX(1)) ) ] ...
                  );

                    
end

