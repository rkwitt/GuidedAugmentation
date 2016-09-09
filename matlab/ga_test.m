% clear all;
% load ('/Volumes/STORE/SUN-RGBD/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat');
% obj_info = ga_extract_original_object_BB(SUNRGBDMeta, 'chair', 0);
% obj_info_pruned = ga_save_fastRCNN(obj_info, '/tmp/chairs','/Volumes/STORE/SUN-RGBD/');
% disp(length(obj_info_pruned));

val = load('/tmp/vae/ae_X_src_val.txt');
res = load('/tmp/vae/ae_X_src_res.txt');
est = val + res;

% X_tst = load('/tmp/vae/X_tst.txt');

% training/testing split for full data
p_trn = load('/tmp/vae/p_trn.txt');
p_tst = load('/tmp/vae/p_tst.txt');

obj_info_trn = obj_info_pruned(p_trn);
obj_info_tst = obj_info_pruned(p_tst);

% source/destination split of training data
p_trn_src = load('/tmp/vae/p_trn_src.txt');
p_trn_dst = load('/tmp/vae/p_trn_dst.txt');

obj_info_trn_src = obj_info_trn(p_trn_src);
obj_info_trn_dst = obj_info_trn(p_trn_dst);

% training/validation indices from source AND destination used by AE training
ae_p_trn = load('/tmp/vae/ae_p_src_trn.txt');
ae_p_val = load('/tmp/vae/ae_p_src_val.txt');

obj_info_trn_src_ae_trn = obj_info_trn_src(ae_p_trn);
obj_info_trn_src_ae_val = obj_info_trn_src(ae_p_val);


for i=100:200
    subplot(1,4,1);
    setenv('SUNRGBD', '/Volumes/STORE/SUN-RGBD/');
    im = imread(fullfile(getenv('SUNRGBD'), ...
        obj_info_trn_src_ae_val(i).img_file));
    imshow(im)
    bb = obj_info_trn_src_ae_val(i).bb;
    rectangle('Position',[bb(1),bb(2),bb(3),bb(4) ], 'EdgeColor','blue', 'LineWidth',3);
    title(sprintf('Query image + depth: %.2f [m]', obj_info_trn_src_ae_val(i).depth));

    retrieval_distances = pdist2(X_tst, est(i,:));
    [sorted_retrieval_distances, sorted_retrieval_distances_idx] = sort(retrieval_distances);
    for j=1:3
        subplot(1,4,1+j);
        im = imread(fullfile(getenv('SUNRGBD'), ...
            obj_info_tst(sorted_retrieval_distances_idx(j)).img_file));
        imshow(im)
        bb = obj_info_tst(sorted_retrieval_distances_idx(j)).bb;
        rectangle('Position',[bb(1),bb(2),bb(3),bb(4) ], 'EdgeColor','blue', 'LineWidth',3);
        title(sprintf('Retrieved image + depth: %.2f [m]', obj_info_tst(sorted_retrieval_distances_idx(j)).depth));
    end
    pause(0.5);
end









