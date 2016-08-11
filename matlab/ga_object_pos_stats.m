function[] = ga_object_pos_stats(query_ob)

%%%%%%%%%%%%%%%%%%%%%%%%%%
% function[] = object_pos_stats(query_ob)
% Plots centroid locations/distance of 3D boxes covering queried objects  
% 
% query_ob: name of object
%

addpath(genpath('.'))
load(fullfile(getenv('SUNRGBD_dir'), './Metadata/SUNRGBDMeta.mat'));


NImages = size(SUNRGBDMeta, 2);

object_position = [];
s = 1;

for n = 1:NImages
    
data = SUNRGBDMeta(n);

nobjs = size(data.groundtruth3DBB, 2);

%disp(data.rbgpath);

for k = 1:nobjs
    
    if(strcmp(data.groundtruth3DBB(k).classname, query_ob))

        object_position(s).Img_indx = n;
        object_position(s).Centroid = data.groundtruth3DBB(k).centroid;
        object_position(s).Depth = norm(object_position(s).Centroid);
    
        s = s+1;
    
    end
end

clear data;

end


disp(['Found ', num2str(s-1), ' ', query_ob, 's']);

%% Plot histogram of centroid distances, scatter plot of positions

obj_dist = zeros(1, size(object_position, 2));
obj_cent = zeros(size(object_position, 2), 3);

for n = 1:size(object_position, 2)
    obj_dist(n) = object_position(n).Depth;
    obj_cent(n, :) = object_position(n).Centroid;
end

figure;
hist(obj_dist, min([100, (s-1)/5]));
title('Histogram of 3D bbox centroid norm/distance from origin');
figure;
scatter3(obj_cent(:, 1), obj_cent(:, 2), obj_cent(:, 3), 'ro');
title('Scatter plot of 3D bbox centroids');
figure;
subplot(2, 2, 1)
hist(obj_cent(:, 1), min([100, (s-1)/5]));
title('Histograms of X coordinates')
subplot(2, 2, 2)
hist(obj_cent(:, 2), min([100, (s-1)/5]));
title('Histograms of Z coordinates')
subplot(2, 2, 3)
hist(obj_cent(:, 3), min([100, (s-1)/5]));
title('Histograms of Y coordinates')

