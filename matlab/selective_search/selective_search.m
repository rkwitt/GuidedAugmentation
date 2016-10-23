%load ('/Volumes/STORE/SUN-RGBD/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat');


image_db = getenv('SUNRGBD_dir');

N = length(SUNRGBDMeta);

image_filenames = cell(N,1);
for i=1:N
    image_filenames{i} = fullfile(image_db, ...
        SUNRGBDMeta(i).sequenceName, 'image', ...
        SUNRGBDMeta(i).rgbname);
end

% run selective search on all images
selective_search_rcnn(image_filenames, 'SUNRGBD-ssbox.mat');