clear; close all; clc;
addpath('functions\ukf');

% read the point cloud from STL/PLY file
ptCloud   = pcread('data/bunny/reconstruction/bun_zipper_res3.ply');
% ptCloud   = pcread('data/happy_recon/happy_vrip_res3.ply');
% ptCloud   = pcread('data/dragon_recon/dragon_vrip_res3.ply');
N_ptCloud = ptCloud.Count;

% prepare Ŭ, the noiseless, complete, moving dataset
scale     = 1000; % to make sure the scale in mm
N_point   = 30;
U_breve   = (ptCloud.Location - mean(ptCloud.Location))' * scale;

% contruct a arbritary transformation then apply it to Ŭ in order to
% generate Y̆, the noiseless, complete, fixed dataset.
max_t        = 20;
max_theta    = 20;
random_trans = -max_t     + (max_t -(-max_t))         .* rand(1, 3);
random_theta = -max_theta + (max_theta -(-max_theta)) .* rand(1, 3);
random_R     = eul2rotm(deg2rad(random_theta), 'ZYX');
GT           = [random_trans, random_theta];
Y_breve      = random_R * U_breve + random_trans';

% add isotropic zero-mean gaussian noise to Y_breve, simulating noise measurement
var_yacute   = 3;
Sigma_yacute = var_yacute * eye(3);
n_yacute     = mvnrnd( [0 0 0], Sigma_yacute, N_point);
% create a random index which will be used to sample Ŭ
rand_idx     = randperm(N_ptCloud, N_point);
% sample Ŭ and give them gaussian noise, simulating real measurement, that
% is, our measurement is always incomplete and subject to noise.
U = U_breve(:,rand_idx) + n_yacute';

%
% plot Ŭ, the noiseless, complete, moving dataset
figure1 = figure(1);
figure1.WindowState  = 'maximized';
axes1 = axes('Parent', figure1);
plot3( axes1, ...
       U_breve(1,:), ...
       U_breve(2,:), ...
       U_breve(3,:), ...
       '.r', 'MarkerSize', 0.1, ...
       'Tag', 'plot_Ubreve');
xlabel('X'); ylabel('Y');
grid(axes1, 'on'); axis(axes1, 'equal'); hold(axes1, 'on');
% plot U, the noisy, incomplete, moving dataset
plot3( axes1, ...
       U(1,:), ...
       U(2,:), ...
       U(3,:), ...
       'or', ...
       'Tag', 'plot_U');
%plot Y̆, the noiseless, complete, fixed dataset.
plot3( axes1, ...
       Y_breve(1,:), ...
       Y_breve(2,:), ...
       Y_breve(3,:), ...
       '.b',  'MarkerSize', 0.1, ...
       'Tag', 'plot_Ybreve');
legend('Ŭ (noiseles, complete, moving set)', 'U (noisy, incomplete, moving set)', 'Y̆ (noiseless, complete, fixed dataset)', 'Location', 'Best');
view(axes1, -30,5);

%
   
%% 1) Initialization
% In the paper, they uses subcript i to describe 'time' (or in registration
% case, iteration, since it is invariant with time). I changed the notation
% to k, so that there will be no ambiguity with i (for indexing points
% notation), and with t (for translation notation)

% State vector (Eq. 21) initialized with zero
xest_k    = zeros(6,1);
n         = length(xest_k);
% Covariance of state vector intitialized with identity matrix
Pest_xk   = eye(n);

% Noise of process model. Initially, the covariance matrix of the process
% model is calculated using the initial uncertainty of each transformation
% parameter (section IV.B, the paragraph under Eq. 41)
var_trans = max_t;
var_theta = max_theta;
Sigma_x_k =  diag([ones(1,3) * var_trans, ones(1,3) * var_theta]);

% initialize the parameter for determining sigma points. These parameters
% is described in section II.B
alpha  = 0.5;
beta   = 2;
kappa  = 100;
lambda = alpha*alpha*(n+kappa)-n;

% registration parameters
U_breve_transformed = U_breve;
U_transformed = U;
threshold     = 0.5;
anneal_var    = 1;
anneal_rate   = 0.98;
max_iter      = 100;

% variable to store the overall transformation
T_all = eye(4);

% In this simulation, i will continue the registration even after all the
% points in U is used. I changed the main loop not to loop for all points
% but to loop for registration iteration.
for iter=1:max_iter
    % reset the parameter
    xest_k    = zeros(6,1);
    Pest_xk   = eye(n);
    Sigma_x_k = Sigma_x_k * anneal_var;
    
    %% 2-3) New Point Insertion and Closest Point Operator
    
    % I will continue registration even until it reached the number of
    % points in U. I still follow the framework from the paper, that is
    % incrementally adding more points to UKF-based registration, however 
    % i need to stop incrementing after it reached the number points in U.
    % Which means, all points in U will be always used after that.
    if(iter<N_point)
        point=iter;
    end
    
    % U_k (capital U) is all point selected until this iteration. Later,
    % there will be u_k (lowercase u), which will be incrementally added
    % 1-by-1 until the number of U_k if UKF framework.
    rand_idx = randperm(N_point, point);
    U_k      = U_transformed(:, rand_idx(1:point));

    % In this simulation, the correspondences is unknown. Closest point
    % operator is used. Find nearest neighbor in Y_breve for all query in U_k
    [idx, dist] = knnsearch(Y_breve', U_k');
    Y_k         = Y_breve(:, idx);
    mean_dist   = mean(dist);    

    % uncomment this if you want to see the point pair
    %{
    delete(findobj('Tag', 'plot_pointpair'));
    for pointpair=1:k
        uk_yk = [u_k(:,pointpair), y_k(:,pointpair)];
        plot3(uk_yk(1,:), uk_yk(2,:), uk_yk(3,:), '-m', 'Tag', 'plot_pointpair');
    end
    %}    
    
    % UKF begins here
    for k=1:point
        %% 4.a.) State Vector Prediction

        % for convenience
        xest_kmin1    = xest_k;
        Pest_xkmin1   = Pest_xk;

        % Compute sigma points. Technically, process model, f(x), is a linear 
        % function (see Eq. 22 and Eq 28), we actually don't need to calculate 
        % sigma points in the first place. But later, our observation model, 
        % h(x), (see Eq. 26) is a non-linear function. So we need to calculate
        % sigma points regardless.
        [xtilde_k, w_m, w_c] = compute_sigma_points(xest_kmin1, Pest_xkmin1, lambda, alpha, beta);

        % Transform sigma points using process model. But, in the paper, there
        % is no transformation, our new state model is from our previous state
        % model but with noise addition of process model. So just skip it.
        % sigma_points_trans = f_transform(sigma_points);

        % Recover the mu and sigma of the transformed sigma points (whereas
        % there is no transformation), so just return back to our original mu
        % and sigma with addition noise (Eq. 28 & 29)
        [xest_bar_k, Pest_bar_xk] = recover_gaussian(xtilde_k, w_m, w_c);
        Pest_bar_xk = Pest_bar_xk + Sigma_x_k;

        %% 4.b.) State Vector Update
        
        % Incrementally adding points to u_k from U_k (see the desc. about
        % wtf is u_k and U_k in section (2.3) above.
        u_k = U_k(:,1:k);

        % Use the estimated state and transform it to measurement (Eq.26, Eq.30)
        ytilde_k = h_transform(xtilde_k, u_k);
        % recover gaussian in measurement space
        [yest_bar_k, P_yk] = recover_gaussian(ytilde_k, w_m, w_c);

        % Add P_yk with measurement noise.
        % In this simulation, there is a change in the definition of measurement
        % noise. Measurement noise consists of intrinsic, Sigma_yacute_k 
        % (localization error), and extrinsic, Sigma_int_k (false matches) 
        % error. Refer to Eq. 41.
        % Sigma_int_k is assumed to be fixed for every point in the dataset,
        % and set to be the mean square distance of the matching points 
        % reported by the closest operator (refer to the paragraph below Eq. 41)       
        temp           = repelem({1.25*Sigma_yacute}, k);
        Sigma_yacute_k = blkdiag(temp{:});
        temp           = repelem({10*mean_dist * eye(3)}, k);
        Sigma_int_k    = blkdiag(temp{:});
        Sigma_i        = Sigma_yacute_k + Sigma_int_k;
        P_yk           = P_yk + Sigma_i;

        % compute P_xkyk 
        P_xkyk = zeros(size(xtilde_k,1), size(ytilde_k,1));
        for j=1:size(xtilde_k,2)
            temp   = w_c(j) * (xtilde_k(:,j) - xest_bar_k) * (ytilde_k(:,j) - yest_bar_k)';
            P_xkyk = P_xkyk + temp;
        end

        % compute kalman gain
        K_k = P_xkyk / P_yk;

        % update state vector
        y_k     = reshape( Y_k(:, 1:k), [], 1 );
        xest_k  = xest_bar_k  + K_k * (y_k - yest_bar_k); % (Eq. 31)
        Pest_xk = Pest_bar_xk - K_k * P_yk * K_k';        % (Eq. 32)
    
    end    
    
    %% 5) Dataset Transformation
    % use the estimated transformation parameters (xest_k) to ipdate the
    % moving dataset (U)
    xest_bar_trans   = xest_k(1:3);
    xest_bar_theta   = xest_k(4:6);
    R_xest_bar_theta = eul2rotm(deg2rad(xest_bar_theta'), 'ZYX');
    
    U_transformed       = R_xest_bar_theta * U_transformed + xest_bar_trans;
    U_breve_transformed = R_xest_bar_theta * U_breve_transformed + xest_bar_trans;
    
    T_all = [R_xest_bar_theta, xest_bar_trans; zeros(1,3), 1] * T_all;

    %% 6) Convergence check
    % if E[d^2] (expected/mean distance) is less than certain threshold,
    % stop the algorithm.

    if (mean_dist < threshold)
        break
    else
        % Anneal Sigma_x_k with factor of 0.95
        anneal_var = anneal_var * anneal_rate;
    end
    
    %% (Extra) Plotting
    
    fprintf('%d %d %.4f\n\n', iter, k, mean_dist);
%     disp(GT);
%     disp(xest_total');
    
    delete(findobj('Tag', 'plot_U'));
    delete(findobj('Tag', 'plot_Ubreve'));
    delete(findobj('Tag', 'plot_U_trans'));
    delete(findobj('Tag', 'plot_Ubreve_trans'));
    plot3( axes1, ...
           U_transformed(1,:), ...
           U_transformed(2,:), ...
           U_transformed(3,:), ...
           'om', ...
           'Tag', 'plot_U_trans');
    plot3( axes1, ...
           U_breve_transformed(1,:), ...
           U_breve_transformed(2,:), ...
           U_breve_transformed(3,:), ...
           '.r', ...
           'Tag', 'plot_Ubreve_trans');
      
    drawnow;
    
end

t_all   = T_all(1:3, 4);
R_all   = T_all(1:3, 1:3);
eul_all = rad2deg(rotm2eul(R_all, 'ZYX'));
U_est   = R_all * U_breve + t_all;

figure2 = figure(2);
figure2.WindowState  = 'maximized';
axes2 = axes('Parent', figure2);
plot3( axes2, ...
       Y_breve(1,:), ...
       Y_breve(2,:), ...
       Y_breve(3,:), ...
       '.b',  'MarkerSize', 0.1);
grid on; axis equal; hold on;
plot3( axes2, ...
       U_est(1,:), ...
       U_est(2,:), ...
       U_est(3,:), ...
       '.r',  'MarkerSize', 0.1);

disp(GT);
disp([t_all', eul_all]);
disp( abs(GT - [t_all', eul_all]));
fprintf('True RMSE: %d\n', mean(sqrt(sum((U_est - Y_breve).^2, 2))));
   
   
   
   