base_path = '/Users/rkwitt/Remote/GuidedAugmentation/python/test/';

% testing via retrieval
% load ('/Volumes/STORE/SUN-RGBD/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat');
% obj_info = ga_extract_original_object_BB(SUNRGBDMeta, 'chair', 0);
% obj_info_pruned = ga_save_fastRCNN(obj_info, '/tmp/chairs','/Volumes/STORE/SUN-RGBD/');
% disp(length(obj_info_pruned));

% get activations at target attribute value
X_source_val = hdf5read(fullfile(base_path, 'output.hdf5'), 'X_source_val')';
X_target_est = hdf5read(fullfile(base_path, 'prediction.hdf5'), 'prediction')';

% load testing data = external database used for retrieval experiment
X_tst = hdf5read(fullfile(base_path, 'output.hdf5'), 'X_tst')';

% load indices?
p_trn    = hdf5read(fullfile(base_path, 'output.hdf5'), 'p_trn'); % trn indices (for FULL data)
p_tst    = hdf5read(fullfile(base_path, 'output.hdf5'), 'p_tst'); % tst indices (for FULL data)
p_source = hdf5read(fullfile(base_path, 'output.hdf5'), 'p_source'); % trn indices for source (w.r.t. training data) 
p_target = hdf5read(fullfile(base_path, 'output.hdf5'), 'p_target'); % trn indices for target (w.r.t. training data)
p_source_trn = hdf5read(fullfile(base_path, 'output.hdf5'), 'p_source_trn');
p_source_val = hdf5read(fullfile(base_path, 'output.hdf5'), 'p_source_val');

% get slices of obj_info
obj_info_trn = obj_info_pruned(p_trn); 
obj_info_tst = obj_info_pruned(p_tst);

obj_info_source = obj_info_trn(p_source);
obj_info_target = obj_info_trn(p_target);

obj_info_source_trn = obj_info_source(p_source_trn);
obj_info_source_val = obj_info_source(p_source_val);


for i=1:100
    
    % show query object
    subplot(1,4,1);
    im = imread(obj_info_source_val(i).img_file);
    imshow(im);
    
    % show bounding box of query object
    bb = obj_info_source_val(i).bb;
    rectangle('Position',[bb(1),bb(2),bb(3),bb(4) ], 'EdgeColor','blue', 'LineWidth',3);
    title(sprintf('Query image + depth: %.2f [m]', obj_info_source_val(i).depth));

    % NN retrieval based on estimated object activations
    retrieval_distances = pdist2(X_tst, X_target_est(i,:));
    [sorted_retrieval_distances, sorted_retrieval_distances_idx] = sort(retrieval_distances);

    depths = [];
    for j=1:3
        subplot(1,4,1+j);
        im = imread(obj_info_tst(sorted_retrieval_distances_idx(j)).img_file);
        imshow(im)
        bb = obj_info_tst(sorted_retrieval_distances_idx(j)).bb;
        rectangle('Position',[bb(1),bb(2),bb(3),bb(4) ], 'EdgeColor','blue', 'LineWidth',3);
        title(sprintf('Retrieved image + depth: %.2f [m]', obj_info_tst(sorted_retrieval_distances_idx(j)).depth));
        depths = [depths obj_info_tst(sorted_retrieval_distances_idx(j)).depth];
    end
    disp(mean(depths));
    pause;
end









