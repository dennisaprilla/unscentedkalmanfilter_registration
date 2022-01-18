clear; close all;
addpath(genpath('..\functions\display'));

filename = 'isotropic_v6trials_bunny';
load(strcat(filename,'.mat'));

% rearrange data, the requirement for boxplotGroup, please refer
% https://www.mathworks.com/matlabcentral/answers/331381-how-do-you-create-a-grouped-boxplot-with-categorical-variables-on-the-x-axis#answer_418952
data = {};
for init_pose=1:length(description.init_poses)
    for dof=1:6
        data{dof,init_pose} = reshape( absolute_errors(:, dof, 1:4, init_pose), [], 4);
    end
end

% we use subaxis function to control more for the spacing for the subplot
% https://www.mathworks.com/matlabcentral/fileexchange/3696-subaxis-subplot
figure('Name', 'Translation Error', 'Position', [0 0 1200 900])
total_poses = length(description.init_poses);
for axis=1:(total_poses*2)
    
    init_pose = mod( axis-1, total_poses)+1;    
    subaxis(2,3, axis, 'SpacingVertical',0.15, 'SpacingHorizontal',0); hold on;
    
    boxplotGroup( data(1:3,init_pose)', ...
                  'PrimaryLabels', {'tx', 'ty', 'tz'}, ...
                  'SecondaryLabels', {'0', '1', '2', '3'});
              
    grid on;
    
    if(init_pose==1)
        if(axis<3)
            ylabel('Translation MAD (degree)');
        else
            ylabel('Rotation MAD (degree)');
        end
    else
        set(gca,'yticklabel', [])
    end
    ylim([0, 5]);
    title(sprintf('Initial Pose: %d', description.init_poses(init_pose)));
end

% save the picture
saveas(gcf, sprintf('pictures/%s', filename), 'png');