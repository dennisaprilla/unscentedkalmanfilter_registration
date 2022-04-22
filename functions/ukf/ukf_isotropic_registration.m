function [T_all, mean_dist, history] = ukf_isotropic_registration(moving, fixed, varargin)
% SUMMARY: This is an implementation of Unscented Kalman Filter (UKF)-based 
% 3D point cloud registration (https://ieeexplore.ieee.org/document/4359030).
%
% I implemented this based on their description in their paper.
% Misinterpretation of the equations might happen, but i did my best to
% implement this correctly. PLEASE read the paper first to understand the 
% general idea of how the algorithm works.
%
% They claim that it performs better than ICP in noisy measurements. It
% register incrementally when more and more data is retrieved, so it
% is suitable for medical application. If you want to use this function,
% please cite the original paper.
%
% DA. Christie, University of Twente, 2021.
%
% INPUT:
%
%   moving         A 3xN matrix with each column represent the number of
%                  measurement/observation within the set. This is a point set 
%                  will be moving towards the fixed point set (the target).
%
%   fixed          A 3xM matrix with each column represent the point within 
%                  the set. This is the fixed/target point set.
%
%   movingdebug    A 3xM matrix with each column represent the point within
%                  the set. This is ONLY for debugging purpose. Sometimes 
%                  it is hard to see our measurements (moving) point cloud 
%                  because it is noisy and sparse. So if you have the ground
%                  truth, you can feed them to the function, and you can see
%                  the registration in action. This point set will NOT be 
%                  used inside the algorithm, it only used for display. 
%                  Note: you should specify the "display" argument. 
%
%   alpha          A floating point between 0 and 1. This is parameter for
%                  the UKF which determines the spreadness of the sigma 
%                  points. Check UKF theory to know more about this.
%
%   kappa          An integer. You can specify as high as hundreds. This
%                  is a parameter for the UKF which also determines the
%                  spreadness of the sigma points. Check UKF theory to know
%                  more about this.
%
%   beta           An integer between 1-3. In the original paper of UKF,
%                  they claim 2 is the optimum one for gaussian
%                  distribution. Check UKF theory to know more about this.
%
%   iteration      An integer which specify the maximum iteration for
%                  registration algorithm.
%
%   threshold      A floating point which specify the condition to stop the
%                  registration.
%
%   expectednoise  A floating point which determines the initial
%                  observation model noise for the UKF. In the paper, this
%                  parameter will determine the Sigma_y.
%
%   sigmaxanneal   A floating point between 0-1. This is the annealation
%                  rate for the process model noise. If we put 1, that 
%                  means the noise is stay strong, the state estimation
%                  of UKF might consists a high uncertainty throughout the
%                  iteration. In the paper, they specified as 0.95.
%
%   sigmaxtrans    A floating point which determines the variance of the
%                  translation transformation for process model noise
%                  initialization. In the paper, this parameter refered as
%                  Sigma_x (Upper-left 3x3 matrix of Sigma_x).
%
%   sigmaxtheta    Similar to sigmaxtrans, this will determines the
%                  variance of rotation transformation for process model
%                  noise initialization. In the paper, this parameter
%                  refered as Sigma_x (Lower-right 3x3 matrix of Sigma_x).
%
%   bestrmse       A logical. If this parameter specified, this function
%                  will choose the transformation based on the best mean
%                  distance that is estimated by UKF. This is my addition 
%                  to the function, not from original paper. Check the last
%                  last part of this function.
%
%   verbose        A logical. If you want to see the progress of the
%                  transformation and mean distance between the moving and
%                  fixed point cloud, set this parameter to true.
%
%   display        If you want to display the progress of the registration,
%                  set this parameter to true. If you also want to see the
%                  groundtruth point set transforming towards to fixed
%                  point set, you should specify it the "movingdebug"
%                  parameter.
%
% OUTPUT:
% 
%   T_all          A 4x4 homogeneous tranformation matrix, the overall
%                  result of registration. Rotation is specified as 
%                  T_all(1:3, 1:3), translation is specified as T_all(1:3, 4).
%
%   mean_dist      The mean euclidean distance between the measurement / 
%                  moving point set and the fixed point set.
%
%   history        Sequence of all transformation and mean distance during
%                  the registration loop. A struct consist of two fields,
%                  history.transformations and history.mean_distances

%% 0) Input parser and preparation for the function

p = inputParser;
addRequired( p, 'moving', ...
             @(x) validateattributes(x, {'double'}, {'nrows', 3}) );
addRequired( p, 'fixed', ...
             @(x) validateattributes(x, {'double'}, {'nrows', 3}) );
addOptional( p, 'movingdebug', zeros(3,1), ...
             @(x) validateattributes(x, {'double'}, {'nrows', 3}) );
         
addParameter( p, 'alpha', 0.5, ...
              @(x) isnumeric(x) && x>0 && x<1);
addParameter( p, 'beta', 2, ...
              @(x) isnumeric(x) );
addParameter( p, 'kappa', 100, ...
              @(x) isnumeric(x) );
addParameter( p, 'iteration', 100, ...
              @(x) isnumeric(x) );
addParameter( p, 'threshold', 1, ...
              @(x) isnumeric(x) );
addParameter( p, 'expectednoise', 0, ...
              @(x) isnumeric(x) );
addParameter( p, 'sigmaxanneal', 0.95, ...
              @(x) isnumeric(x) && x>0 && x<1 );
addParameter( p, 'sigmaxtrans', 1, ...
              @(x) isnumeric(x) );
addParameter( p, 'sigmaxtheta', 1, ...
              @(x) isnumeric(x) );

addParameter( p, 'bestrmse', false, ...
              @(x) islogical(x) );
addParameter( p, 'verbose', true, ...
              @(x) islogical(x) );
addParameter( p, 'display', true, ...
              @(x) islogical(x) );
parse(p, moving, fixed, varargin{:});

% renaming the arguments from this function
U_breve = p.Results.movingdebug;
U       = moving;
Y_breve = fixed;
N_point = length(U);

% display results only if the user specify the display argument
if(p.Results.display)
    figure1 = figure('Name', 'UKF Registration');
    figure1.WindowState  = 'maximized';
    axes1 = axes('Parent', figure1);
    xlabel(axes1, 'X'); ylabel(axes1, 'Y'); zlabel(axes1, 'Z');
    grid(axes1, 'on'); axis(axes1, 'equal'); hold(axes1, 'on');
    
    % if user specify movingdebug, plot Ŭ, the noiseless, complete, moving dataset
    if(~any(p.Results.movingdebug))
        plot3( axes1, ...
               U_breve(1,:), ...
               U_breve(2,:), ...
               U_breve(3,:), ...
               '.r', 'MarkerSize', 0.1, ...
               'Tag', 'plot_Ubreve');
    end
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
    legend( 'Ŭ (noiseles, complete, moving set)', ...
            'U (noisy, incomplete, moving set)', ...
            'Y̆ (noiseless, complete, fixed dataset)', ...
            'Location', 'Best');
end

fprintf('Registering...');

%% 1) Initialization
% In the paper, they used subcript i to describe 'time' (or in registration
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
var_trans = p.Results.sigmaxtrans;
var_theta = p.Results.sigmaxtheta;
Sigma_x_k =  diag([ones(1,3) * var_trans, ones(1,3) * var_theta]);

% Noise of observation model. Later, The noise is actually added by Sigma_int
% (the noise for compensating false point matches, see section 4.b in this
% code). I will initialize the Sigma_yacute here.
var_yacute   = p.Results.expectednoise;
Sigma_yacute = var_yacute * eye(3);

% initialize the parameter for determining sigma points. These parameters
% is described in section II.B
alpha  = p.Results.alpha;
beta   = p.Results.beta;
kappa  = p.Results.kappa;
lambda = alpha*alpha*(n+kappa)-n;

% registration parameters
U_breve_transformed = U_breve;
U_transformed = U;
threshold     = p.Results.threshold;
anneal_var    = 1;
anneal_rate   = p.Results.sigmaxanneal;
max_iter      = p.Results.iteration;

% variable to store the overall transformation
T_all = eye(4);
% variable to store the tansformations and mean distance history
mean_distances  = [];
transformations = [];

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
    
    % transform the U
    U_transformed       = R_xest_bar_theta * U_transformed + xest_bar_trans;
    U_breve_transformed = R_xest_bar_theta * U_breve_transformed + xest_bar_trans;
    
    % store the transformation
    T_all = [R_xest_bar_theta, xest_bar_trans; zeros(1,3), 1] * T_all;
    
    % store the transformation history
    t_all   = T_all(1:3, 4);
    R_all   = T_all(1:3, 1:3);
    eul_all = rad2deg(rotm2eul(R_all, 'ZYX'));
    transformations = [transformations; t_all', eul_all];
    
    % store the mean distance history
    mean_distances  = [mean_distances, mean_dist];

    %% 6) Convergence check
    
    % if E[d^2] (expected/mean distance) is less than certain threshold,
    % stop the algorithm.
    if (mean_dist < threshold)
        break
    else
        % Anneal Sigma_x_k with factor of anneal_rate
        anneal_var = anneal_var * anneal_rate;
    end
    
    %% (Extra) Plotting
    
    if (p.Results.verbose)
        % print current iteration, point used, and mean distance
        fprintf('%d %d %.4f\n\n', iter, k, mean_dist);
        
        % print current estimated transformation
        disp([t_all', eul_all]);
    end
    
    if (p.Results.display)
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
    
end

% save the transformation and mean distance history to a struct
history.transformations = transformations;
history.mean_distances  = mean_distances;

%% 7) Search for the best transformation
% This is addition step i write. At first few iteration, UKF drastically
% minimize the RMSE, but at some point of iteration, it stop minimizing,
% and jiggling around some number of RMSE. I observed, there is no 
% guarantee that the last iteration will be the best. So here, i will 
% search the most minimum mean distance of all of the history, and this 
% function will spit out that particular transformation rather than the 
% last transformation.

% if the user specifed 'bestrmse'
if (p.Results.bestrmse)
    % search the minimum mean distance
    [best_meandist_val, best_meandist_idx] = min(mean_distances(N_point+1:end));
    
    % replace the returned mean_dist value to the best one
    mean_dist           = best_meandist_val;
    % replace the returned T_all to the best one according to mean distance
    best_transformation = transformations(N_point+best_meandist_idx, :);
    xest_bar_trans      = best_transformation(1:3);
    xest_bar_theta      = best_transformation(4:6);
    R_xest_bar_theta    = eul2rotm(deg2rad(xest_bar_theta), 'ZYX');
    T_all               = [R_xest_bar_theta, xest_bar_trans'; zeros(1,3), 1];
end

% test
history.chosenrmse = N_point+best_meandist_idx;

fprintf('Finished');

end

