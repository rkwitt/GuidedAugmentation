%load ('/Volumes/STORE/SUN-RGBD/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat');

%load('SUNRGBD-ssbox.mat');

image_db = getenv('SUNRGBD_dir');
out_dir = '/tmp/images';

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

N = length(SUNRGBDMeta);

list_file = '/tmp/allimgs.txt';

fid = fopen(list_file, 'w');

image_filenames = cell(N,1);
for i=1:N
    image_filenames{i} = fullfile(image_db, ...
        SUNRGBDMeta(i).sequenceName, 'image', ...
        SUNRGBDMeta(i).rgbname);
    
    img_file = 'image_%.5d';
    out_file = sprintf(img_file, i);
    fprintf(fid, '%s\n', out_file);
    
    [success,~,~] = copyfile(image_filenames{i}, fullfile(out_dir, ...
        [out_file, '.jpg']));
    disp(success);
    
    box_file = [sprintf('%s', out_file) '_ss_boxes.mat'];
    boxes = all_boxes{i};
    save(fullfile(out_dir, box_file), 'boxes');
    
    
end

fclose(fid);