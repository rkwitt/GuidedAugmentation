load ~/Downloads/chair_data_VOC.mat
load '/Volumes/STORE/SUN-RGBD/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat'

rng(1243);
imgs = chair_attr;
out_dir = '/Users/rkwitt/Remote/GuidedAugmentation/data';

%--------------------------------------------------------------------------
% get the number of images + detections/image
%--------------------------------------------------------------------------
det_cnt = 0;
img_cnt = 0;
for i=1:length(imgs)
    if imgs(i).label == 1
        img_cnt = img_cnt + 1;
        det_cnt = det_cnt + length(imgs(i).depth);
    end
end
fprintf('[%s]: found %d detections [%d images]\n', ...
    datestr(datetime('now')), det_cnt, img_cnt); 


%--------------------------------------------------------------------------
% get the indices of the objects wrt the original images
%--------------------------------------------------------------------------
cnt = 1;
idx = zeros(img_cnt, 1);
for i=1:length(imgs)
    if imgs(i).label == 1 % FOUND
        idx(cnt) = i;
        cnt = cnt + 1;
    end
end


%--------------------------------------------------------------------------
% split data into external/testing images & get indices wrt original images
%--------------------------------------------------------------------------
[ext, tst] = crossvalind('HoldOut', img_cnt, 0.3); % split 70/30
idx_ext = idx(ext); % training indices
idx_tst = idx(tst); % testing indices
N_ext = length(idx_ext);
N_tst = length(idx_tst);
assert(img_cnt == N_ext + N_tst);
fprintf('[%s]: leaving out %d/%d images for testing\n', ...
    datestr(datetime('now')), length(idx_tst), img_cnt);


%--------------------------------------------------------------------------
% split external data into training/validation
%--------------------------------------------------------------------------
[trn, val] = crossvalind('HoldOut', N_ext, 0.5);
idx_trn = idx_ext(trn);
idx_val = idx_ext(val);
N_trn = length(idx_trn);
N_val = length(idx_val);
assert(N_ext == N_trn + N_val);
fprintf('[%s]: using %d images for training \n', ...
    datestr(datetime('now')), N_trn);
fprintf('[%s]: using %d images for validation\n', ...
    datestr(datetime('now')), N_val);


%--------------------------------------------------------------------------
% construct testing data
%--------------------------------------------------------------------------
c = 0;
for i=1:length(idx_tst)
    j = idx_tst(i);
    c = c + length(imgs(j).depth);
end
X_tst = zeros(c, 4096);
y_tst = zeros(c, 1);
c=1;
for i=1:length(idx_tst)
    j = idx_tst(i);
    X_tst(c:c+length(imgs(j).depth)-1,:) = imgs(j).feat;
    y_tst(c:c+length(imgs(j).depth)-1,:) = imgs(j).depth;
    c = c + length(imgs(j).depth);
end


%--------------------------------------------------------------------------
% construct training + validation data
%--------------------------------------------------------------------------
c = 0;
for i=1:length(idx_trn)
    j = idx_trn(i);
    c = c + length(imgs(j).depth);
end
X_trn = zeros(c, 4096);
y_trn = zeros(c, 1);
c = 1;
for i=1:length(idx_trn)
    j = idx_trn(i);
    X_trn(c:c+length(imgs(j).depth)-1,:) = imgs(j).feat;
    y_trn(c:c+length(imgs(j).depth)-1,:) = imgs(j).depth;
    c = c + length(imgs(j).depth);
end
c = 0;
for i=1:length(idx_val)
    j = idx_val(i);
    c = c + length(imgs(j).depth);
end
X_val = zeros(c, 4096);
y_val = zeros(c, 1);
c = 1;
for i=1:length(idx_val)
    j = idx_val(i);
    X_val(c:c+length(imgs(j).depth)-1,:) = imgs(j).feat;
    y_val(c:c+length(imgs(j).depth)-1,:) = imgs(j).depth;
    c = c + length(imgs(j).depth);
end
fprintf('[%s]: extracted %d activations (trn)\n', ...
    datestr(datetime('now')), length(y_trn));
fprintf('[%s]: extracted %d activations (val)\n', ...
    datestr(datetime('now')), length(y_val));


%--------------------------------------------------------------------------
% finally, construct source/target training data
%--------------------------------------------------------------------------

sources = [...
    0.0 1.0; ...
    0.5 1.5; ...
    1.0 2.0; ...
    1.5 2.5; ...
    2.0 3.0; ...
    2.5 3.5; ...
    3.0 4.0; ...
    3.5 4.5; ...
    4.0 5.0];
targets = [...
    1.0 2.0; ... % 2.18, 2.11
    1.5 2.5; ... % 2.21, 2.17
    2.0 3.0; ... % 2.47, 2.45
    2.5 3.5; ... % 2.84, 2.88
    3.0 4.0; ... % 3.63, 3.60
    3.5 4.5; ... % 3.78, 3.80
    4.0 5.0; ...
    4.5 5.5; ...
    5.0 6.0];
assert(size(sources,1)==size(targets,1));

for m=1:size(sources,1)
    
    source_beg = sources(m,1);
    source_end = sources(m,2);
    target_beg = targets(m,1);
    target_end = targets(m,2);
    

    p0 = find(y_trn>=source_beg & y_trn<source_end);
    p1 = find(y_trn>=target_beg & y_trn<target_end);
    p2 = find(y_val>=source_beg & y_val<source_end);
    
    if length(p0) <= 500
        continue;
    end
    if length(p1) <= 500
        continue;
    end
    if length(p2) <= 500
        continue;
    end

    X_source_trn = X_trn(p0,:);
    X_target_tmp = X_trn(p1,:);
    X_source_val = X_trn(p2,:);

    fprintf('[%s]: X_source_trn = (%d x %d)\n', ...
        datestr(datetime('now')), ...
        size(X_source_trn, 1), ...
        size(X_source_trn, 2));
    fprintf('[%s]: X_target_tmp = (%d x %d)\n', ...
        datestr(datetime('now')), ...
        size(X_target_tmp, 1), ...
        size(X_target_tmp, 2));
    fprintf('[%s]: X_source_val = (%d x %d)\n', ...
        datestr(datetime('now')), ...
        size(X_source_val, 1), ...
        size(X_source_val, 2));

    %----------------------------------------------------------------------
    % pair source activations with closest neighbor in target
    %----------------------------------------------------------------------

    tmp = pdist2(X_source_trn, X_target_tmp);
    [sd,si] = sort(tmp, 2);

    X_target_trn = zeros(size(X_source_trn));
    for i=1:size(X_source_trn,1)
        X_target_trn(i,:) = X_target_tmp(si(i,1),:);
    end

    trn_file = fullfile(out_dir, sprintf('train_%d.hdf5', m));
    val_file = fullfile(out_dir, sprintf('validation_%d.hdf5', m));
    
    hdf5write(trn_file, ...
        'X_source_trn', X_source_trn, ...
        'X_target_trn', X_target_trn);
    hdf5write(val_file, ...
        'X_source_val', X_source_val);
end

hdf5write(fullfile(out_dir, 'test.hdf5'), ...
    'X_tst', X_tst, ...
    'y_tst', y_tst);


% DEBUG
tmp = zeros(500,6);
for m=1:6
    prediction_file = fullfile(out_dir, sprintf('prediction_%d.hdf5', m));
    x = hdf5read(prediction_file, 'prediction')';
    p = randi(size(x,1),500,1);
    K = pdist2(x(p,:), X_tst); 
    [~,xi] = sort(K,2); 
    tmp(:,m) = y_tst(xi(:,1));
    avg = mean(tmp(:,m));
    dev =  std(tmp(:,m));
    disp(avg);
    disp(dev);
end
















