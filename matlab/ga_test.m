% testing via retrieval
% load ('/Volumes/STORE/SUN-RGBD/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat');
% obj_info = ga_extract_original_object_BB(SUNRGBDMeta, 'chair', 0);
% obj_info_pruned = ga_save_fastRCNN(obj_info, '/tmp/chairs','/Volumes/STORE/SUN-RGBD/');
% disp(length(obj_info_pruned));

% load data
X_tst = hdf5read('/Users/rkwitt/Remote/GuidedAugmentation/python/output.hdf5', 'X_tst');
X_val = hdf5read('/Users/rkwitt/Remote/GuidedAugmentation/python/output.hdf5', 'ae_X_src_val');
X_res = hdf5read('/Users/rkwitt/Remote/GuidedAugmentation/python/output.hdf5', 'ae_X_src_res');

X_tst = X_tst';
X_val = X_val';
X_res = X_res';

X_est = single(X_res); % estimated representations at target
X_tst = single(X_tst);

% load indices
p_trn = hdf5read('/tmp/vae/output.hdf5', 'p_trn'); % trn indices (for FULL data)
p_tst = hdf5read('/tmp/vae/output.hdf5', 'p_tst'); % tst indices (for FULL data)
p_trn_src = hdf5read('/tmp/vae/output.hdf5', 'p_trn_src'); % trn indices for SRC (for training data) 
p_trn_dst = hdf5read('/tmp/vae/output.hdf5', 'p_trn_dst'); % trn indices for DST (for training data)
ae_p_src_trn = hdf5read('/Users/rkwitt/Remote/GuidedAugmentation/python/output.hdf5', 'ae_p_src_trn'); % trn indices used for AE source
ae_p_src_val = hdf5read('/Users/rkwitt/Remote/GuidedAugmentation/python/output.hdf5', 'ae_p_src_val'); % trn indices used for AE validation

obj_info_trn = obj_info_pruned(p_trn); 
obj_info_tst = obj_info_pruned(p_tst);

obj_info_trn_src = obj_info_trn(p_trn_src);
obj_info_trn_dst = obj_info_trn(p_trn_dst);

obj_info_trn_src_ae_trn = obj_info_trn_src(ae_p_src_trn);
obj_info_trn_src_ae_val = obj_info_trn_src(ae_p_src_val);


for i=1:100
    
    % show query object
    subplot(1,4,1);
    im = imread(obj_info_trn_src_ae_val(i).img_file);
    imshow(im)
    
    bb = obj_info_trn_src_ae_val(i).bb;
    rectangle('Position',[bb(1),bb(2),bb(3),bb(4) ], 'EdgeColor','blue', 'LineWidth',3);
    title(sprintf('Query image + depth: %.2f [m]', obj_info_trn_src_ae_val(i).depth));

    % get retrieval results with estimated object activations
    retrieval_distances = pdist2(X_tst, X_est(i,:));
    [sorted_retrieval_distances, sorted_retrieval_distances_idx] = sort(retrieval_distances);

    % plot retrieval results
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









