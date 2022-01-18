clear; clf; clc;
addpath('functions\ukf');

N = 30;
% generate U_breve, the moving dataset
% U_breve = [ linspace(-250, 250, N); linspace(-250, 250, N); linspace(-250, 250, N) ];
% U_breve = [ linspace(-250, 250, N); linspace(-250, 250, N); zeros(1, N) ];
U_breve = [ linspace(-250, 250, N); zeros(2, N) ];

% contruct a arbritary transformation to generate Y_breve, the fixed dataset
max_t        = 10;
max_theta    = 10;
random_trans = -max_t     + (max_t -(-max_t))         .* rand(1, 3);
random_theta = -max_theta + (max_theta -(-max_theta)) .* rand(1, 3);
% random_R     = rotx(random_theta(1)) * roty(random_theta(2)) * rotz(random_theta(3));
% random_R     = rotx(random_theta(1)+random_theta(3)) * roty(random_theta(2));
random_R     = eul2rotm(deg2rad(random_theta), 'ZYX');
Y_breve      = random_R * U_breve + random_trans';

  
% add isotropic zero-mean gaussian noise to Y_breve, simulating noise measurement
var_yacute   = 3;
Sigma_yacute = var_yacute*eye(3);
n_yacute     = mvnrnd( [0 0 0], Sigma_yacute, N);
Y            = Y_breve + n_yacute';

%
% plot U_breve
figure1 = figure(1);
% figure1.WindowState  = 'maximized';
axes1 = axes('Parent', figure1);
plot3( axes1, ...
       U_breve(1,:), ...
       U_breve(2,:), ...
       U_breve(3,:), ...
       'or', 'MarkerFaceColor', 'r', ...
       'Tag', 'plot_gt');
xlabel('X'); ylabel('Y');
grid(axes1, 'on'); axis(axes1, 'equal'); hold(axes1, 'on');
% plot Y_breve
plot3( axes1, ...
       Y_breve(1,:), ...
       Y_breve(2,:), ...
       Y_breve(3,:), ...
       'ob', 'MarkerFaceColor', 'b', ...
       'Tag', 'plot_gt');
% plot Y_breve
plot3( axes1, ...
       Y(1,:), ...
       Y(2,:), ...
       Y(3,:), ...
       'ob', ...
       'Tag', 'plot_gt');
%


%% 1) Initialization
% In the paper, they uses subcript i to describe 'time' (or in registration
% case, iteration, since it is invariant with time). I changed the notation
% to k, so that there will be no ambiguity with i (for indexing points
% notation), and with t (for translation notation)

% state vector
xest_k = zeros(6,1);
n      = length(xest_k);
% covariance of state vector
Pest_xk = eye(n);
% noise of process model (Eq. 27)
c = sqrt(180);
lambda_for_SigmaX = var(Y, [], 2);
var_for_SigmaX    = diag(Sigma_yacute);
Sigma_x_k         = diag( [ var_for_SigmaX(1), var_for_SigmaX(2), var_for_SigmaX(3), ...
                            c*c / ( sqrt(lambda_for_SigmaX(2)/var_for_SigmaX(3)) + sqrt(lambda_for_SigmaX(3)/var_for_SigmaX(2)) ) , ...
                            c*c / ( sqrt(lambda_for_SigmaX(1)/var_for_SigmaX(3)) + sqrt(lambda_for_SigmaX(3)/var_for_SigmaX(1)) ) , ...
                            c*c / ( sqrt(lambda_for_SigmaX(1)/var_for_SigmaX(2)) + sqrt(lambda_for_SigmaX(2)/var_for_SigmaX(1)) ) ] ...
                        );

% initialize the parameter for determining sigma points. These parameters
% is described in section II.B
alpha  = 0.25;
beta   = 3;
kappa  = 500;
lambda = alpha*alpha*(n+kappa)-n;

GT = [random_trans, random_theta];

for k=1:N
    %% 2) State Vector Prediction

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
    
    %% 3) New point set insertion
    
    % As the paper describes in section III.C, under Fig. 3, they
    % incrementally adding more points in each of iterations. From their
    % first simulation, from Fig. 4, it can be assumed that they iterate 
    % the UKF from 1 until N (number of points).
    % One more thing, in their first simulation, it is assumed that the
    % pair correspondences is known. 
    u_k = U_breve(:,1:k);
    % u_k = U_breve;
    
    % Use the estimated state and transform it to measurement (Eq.26, Eq.30)
    ytilde_k = h_transform(xtilde_k, u_k);
    % recover gaussian in measurement space
    [yest_bar_k, P_yk] = recover_gaussian(ytilde_k, w_m, w_c);
    
    %% 4) State Vector Update    
    
    % add P_yk with measurement noise. It is described under Eq. 26.
    temp           = repelem({Sigma_yacute}, k);
    % temp           = repelem({Sigma_yacute}, N);
    Sigma_yacute_k = blkdiag(temp{:});
    P_yk           = P_yk + Sigma_yacute_k;
    
    % compute P_xkyk 
    P_xkyk = zeros(size(xtilde_k,1), size(ytilde_k,1));
    for j=1:size(xtilde_k,2)
        temp   = w_c(j) * (xtilde_k(:,j) - xest_bar_k) * (ytilde_k(:,j) - yest_bar_k)';
        P_xkyk = P_xkyk + temp;
    end
    
    % compute kalman gain
    K_k = P_xkyk / P_yk;
    
    % update state vector
    y_k     = reshape( Y(:, 1:k), [], 1 );
    % y_k     = reshape( Y, [], 1 );
    xest_k  = xest_bar_k  + K_k * (y_k - yest_bar_k); % (Eq. 31)
    Pest_xk = Pest_bar_xk - K_k * P_yk * K_k';        % (Eq. 32)
    
    %% (Extra) Plotting
    
    % break;
    disp(k);
    disp(GT);
    disp(xest_k');
    
	xest_bar_trans = xest_k(1:3);
    xest_bar_theta = xest_k(4:6);
    % R_xest_bar_theta = rotx(xest_bar_theta(1)+xest_bar_theta(3)) * roty(xest_bar_theta(2));
    R_xest_bar_theta = eul2rotm(deg2rad(xest_bar_theta'), 'ZYX');
    U_breve_transformed = R_xest_bar_theta * U_breve + xest_bar_trans;
    
    delete(findobj('Tag', 'plot_Ubreve'));
    plot3( axes1, ...
           U_breve_transformed(1,:), ...
           U_breve_transformed(2,:), ...
           U_breve_transformed(3,:), ...
           'om', 'MarkerFaceColor', 'm', ...
           'Tag', 'plot_Ubreve');
      
    drawnow;
    
end


% test1 = R_GT * U_breve;
% test2 = R_xest * U_breve;
% figure(2);
% plot3(test1(1,:), test1(2,:), test1(3,:), 'or', 'MarkerFaceColor', 'r');
% grid on; hold on; axis equal;
% plot3(test2(1,:), test2(2,:), test2(3,:), 'ob', 'MarkerFaceColor', 'b');

% % http://www.boris-belousov.net/2016/12/01/quat-dist/
% R = R_GT * R_xest';
% tetha = (acos(trace(R)-1)/2) * (180/pi)



