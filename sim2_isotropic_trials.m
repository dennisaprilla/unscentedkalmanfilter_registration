clear; close all; clc;
addpath('functions\ukf');

% read the point cloud from STL/PLY file
ptCloud   = pcread('data/bunny/reconstruction/bun_zipper_res3.ply');
% ptCloud   = pcread('data/happy_recon/happy_vrip_res3.ply');
% ptCloud   = pcread('data/dragon_recon/dragon_vrip_res3.ply');
N_ptCloud = ptCloud.Count;

% prepare Ŭ, the noiseless, complete, moving dataset
scale     = 1000; % to make sure the scale in mm
U_breve   = (ptCloud.Location - mean(ptCloud.Location))' * scale;

%% Simulation trials

number_trials     = 500;
absolute_errors   = zeros(number_trials, 6);
rmse_measurements = zeros(number_trials, 1);
rmse_trues        = zeros(number_trials, 1);

max_t      = 10;
max_theta  = 10;
N_point    = 30;
var_yacute = 2;

for trial=1:number_trials
    disp(trial);
    
    %% radom transformation, point selection, and noise
    
    % contruct a arbritary transformation then apply it to Ŭ in order to
    % generate Y̆, the noiseless, complete, fixed dataset.
    random_trans = -max_t     + (max_t -(-max_t))         .* rand(1, 3);
    random_theta = -max_theta + (max_theta -(-max_theta)) .* rand(1, 3);
    random_R     = eul2rotm(deg2rad(random_theta), 'ZYX');
    GT           = [random_trans, random_theta];
    Y_breve      = random_R * U_breve + random_trans';

    % add isotropic zero-mean gaussian noise to Y_breve, simulating noise measurement
    Sigma_yacute = var_yacute * eye(3);
    n_yacute     = mvnrnd( [0 0 0], Sigma_yacute, N_point);
    % create a random index which will be used to sample Ŭ
    rand_idx     = randperm(N_ptCloud, N_point);
    % sample Ŭ and give them gaussian noise, simulating real measurement, that
    % is, our measurement is always incomplete and subject to noise.
    U = U_breve(:,rand_idx) + n_yacute';

    %% registration

    [T_all, mean_dist] = ukf_isotropic_registration( U, Y_breve, U_breve, ...
                           'threshold', 0.5, ...
                           'iteration', 120, ...
                           'expectednoise', 1.25*var_yacute, ...
                           'sigmaxanneal', 0.98, ...
                           'sigmaxtrans', max_t, ...
                           'sigmaxtheta', max_theta, ...
                           'verbose', false, ...
                           'display', false);

    %% calculate performance

    t_all      = T_all(1:3, 4);
    R_all      = T_all(1:3, 1:3);
    eul_all    = rad2deg(rotm2eul(R_all, 'ZYX'));
    Uest_breve = R_all * U_breve + t_all;

    absolute_errors(trial, :) = abs(GT - [t_all', eul_all]);
    rmse_measurements(trial)  = mean_dist;
    rmse_trues(trial)         = mean(sqrt(sum((Uest_breve - Y_breve).^2, 2)));

end

% display boxplot
figure;
subplot(2, 1, 1);
boxplot(absolute_errors(:, 1:3), {'dtx', 'dty', 'dtz'});
title('Rotation');
xlabel('Degree of Freedom');
ylabel('Abs. Difs. (degree)');
ylim([0 10]);
grid on;
subplot(2,1, 2);
boxplot(absolute_errors(:, 4:6), {'dRz', 'dRy', 'dRx'});
title('Translation');
xlabel('Degree of Freedom');
ylabel('Abs. Difs. (mm)');
ylim([0 10]);
grid on;

figure;
histogram(rmse_measurements, 100);
title('RMSE measurement');
xlabel('Mean Error');
ylabel('Counts');
grid on;

figure;
histogram(rmse_trues, 100);
title('RMSE true');
xlabel('Mean Error');
ylabel('Counts');
grid on;







   