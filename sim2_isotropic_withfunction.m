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
max_t        = 8;
max_theta    = 8;
random_trans = -max_t     + (max_t -(-max_t))         .* rand(1, 3);
random_theta = -max_theta + (max_theta -(-max_theta)) .* rand(1, 3);
random_R     = eul2rotm(deg2rad(random_theta), 'ZYX');
GT           = [random_trans, random_theta];
Y_breve      = random_R * U_breve + random_trans';

% add isotropic zero-mean gaussian noise to Y_breve, simulating noise measurement
var_yacute   = 2;
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
                       'verbose', true, ...
                       'display', true);

%%

t_all   = T_all(1:3, 4);
R_all   = T_all(1:3, 1:3);
eul_all = rad2deg(rotm2eul(R_all, 'ZYX'));
Uest_breve = R_all * U_breve + t_all;
Uest       = R_all * U + t_all;

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
       Uest_breve(1,:), ...
       Uest_breve(2,:), ...
       Uest_breve(3,:), ...
       '.r');
plot3( axes2, ...
       Uest(1,:), ...
       Uest(2,:), ...
       Uest(3,:), ...
       'or');

disp('Results:');
disp(GT);
disp([t_all', eul_all]);
disp( abs(GT - [t_all', eul_all]));
fprintf('Measurement RMSE: %d\n', mean_dist);
fprintf('True RMSE: %d\n', mean(sqrt(sum((Uest_breve - Y_breve).^2, 2))));
   
   
   
   