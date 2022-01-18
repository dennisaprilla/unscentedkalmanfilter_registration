clear; close all;
addpath('..\functions\display\subaxis');

filename = 'isotropic_v6trials_bunny';
load(strcat(filename,'.mat'));

mean_all = mean(absolute_errors, 1);
std_all = std(absolute_errors, 1);

figure('Name', 'Translation Error', 'Position', [0 0 1200 400])
for init_pose=1:length(description.init_poses)
    subaxis(1,3, init_pose, 'Spacing', 0, 'MarginTop', 0.1, 'MarginBottom', 0.2); hold on;
    mean_currentpose = squeeze(mean_all(1 , 1:3, :, init_pose))';
    std_currentpose  = squeeze(std_all(1 , 1:3, :, init_pose))';
    
    handle_bar = bar(description.noises, mean_currentpose);
    drawnow;
    
    for k = 1:3
        % get x positions per group
        a = get(handle_bar, 'XData');
        b = get(handle_bar, 'XOffset');
        offset = a{k}+b{k};
        % draw errorbar
        errorbar(offset, mean_currentpose(:,k), zeros(size(std_currentpose(:,k))), std_currentpose(:,k), 'LineStyle', 'none', 'Color', 'k', 'Marker', 'o');
    end
    
    grid on;
    
    xlabel('Noise Level (mm)');
    if(init_pose==1)
        ylabel('Translation MAD (mm)'); 
    else
        set(gca,'yticklabel', [])
    end
    ylim([0, 5]);
    legend('tx', 'ty', 'tz', 'Location', 'northwest');
    title(sprintf('Initial Pose: %d', description.init_poses(init_pose)));
end

figure('Name', 'Rotation Error', 'Position', [0 0 1200 400])
for init_pose=1:length(description.init_poses)
    subaxis(1,3, init_pose, 'Spacing', 0, 'MarginTop', 0.1, 'MarginBottom', 0.2); hold on;
    mean_currentpose = squeeze(mean_all(1 , 4:6, :, init_pose))';
    std_currentpose  = squeeze(std_all(1 , 4:6, :, init_pose))';
    
    handle_bar = bar(description.noises, mean_currentpose);
    drawnow;
    
    for k = 1:3
        % get x positions per group
        a = get(handle_bar, 'XData');
        b = get(handle_bar, 'XOffset');
        offset = a{k}+b{k};
        % draw errorbar
        errorbar(offset, mean_currentpose(:,k), zeros(size(std_currentpose(:,k))), std_currentpose(:,k), 'LineStyle', 'none', 'Color', 'k', 'Marker', 'o');
    end
    
    grid on;
    
    xlabel('Noise Level (mm)');
    if(init_pose==1)
        ylabel('Rotation MAD (degree)'); 
        legend('Rx', 'Ry', 'Rz', 'Location', 'northwest');
    else
        set(gca,'yticklabel', [])
    end
    ylim([0, 5]);
    title(sprintf('Initial Pose: %d', description.init_poses(init_pose)));
end
