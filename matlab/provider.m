function data = provider(obj_mat, mat_field, field, obj_prefix, seed)

    %----------------------------------------------------------------------
    % CONFIG
    %----------------------------------------------------------------------
    rng(seed);              % set RNG seed for reproducibility
    tmp = load(obj_mat);
    imgs = eval(sprintf('tmp.%s', mat_field));
    out_dir = fullfile('/Users/rkwitt/Remote/GuidedAugmentation/data', obj_prefix);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    %----------------------------------------------------------------------
    % get the number of images + detections/image
    %----------------------------------------------------------------------
    det_cnt = 0;
    img_cnt = 0;
    for i=1:length(imgs)
        if imgs(i).label == 1
            img_cnt = img_cnt + 1;
            det_cnt = det_cnt + length(eval(sprintf('imgs(%d).%s',i, field)));
        end
    end
    fprintf('[%s]: found %d detections [in %d images]\n', ...
        datestr(datetime('now')), det_cnt, img_cnt); 

    %----------------------------------------------------------------------
    % get the indices of the objects wrt the original images
    %----------------------------------------------------------------------
    cnt = 1;
    idx = zeros(img_cnt, 1);
    for i=1:length(imgs)
        if imgs(i).label == 1 % object present in image
            idx(cnt) = i;
            cnt = cnt + 1;
        end
    end

    %----------------------------------------------------------------------
    % split data into external/test. images & get indices wrt original imgs
    %----------------------------------------------------------------------
    [ext, tst] = crossvalind('HoldOut', img_cnt, 0.3);
    idx_ext = idx(ext); % indices of external images
    idx_tst = idx(tst); % indices of testing images
    N_ext = length(idx_ext);
    N_tst = length(idx_tst);
    assert(img_cnt == N_ext + N_tst);
    fprintf('[%s]: separating %d images as external data\n', ...
        datestr(datetime('now')), N_ext);
    fprintf('[%s]: separating %d images for testing\n', ...
        datestr(datetime('now')), N_tst);


    %----------------------------------------------------------------------
    % split external data into training/validation
    %----------------------------------------------------------------------
    [trn, val] = crossvalind('HoldOut', N_ext, 0.5);
    idx_trn = idx_ext(trn); % indices of images that can be used for training
    idx_val = idx_ext(val); % indices of images that can be used for validation
    N_trn = length(idx_trn);
    N_val = length(idx_val);
    assert(N_ext == N_trn + N_val);
    fprintf('[%s]: separating %d images (from external data) for training \n', ...
        datestr(datetime('now')), N_trn);
    fprintf('[%s]: separating %d images (from external data) for validation\n', ...
        datestr(datetime('now')), N_val);


    %----------------------------------------------------------------------
    % construct testing data
    %----------------------------------------------------------------------
    c = 0;
    for i=1:length(idx_tst)
        j = idx_tst(i);
        c = c + length(eval(sprintf('imgs(%d).%s', j, field)));
    end
    X_tst = zeros(c, 4096);
    y_tst = zeros(c, 1);
    I_tst = zeros(c, 1);
    c=1;
    for i=1:length(idx_tst)
        j = idx_tst(i);
        X_tst(c:c+length(eval(sprintf('imgs(%d).%s',j, field)))-1,:) = imgs(j).feat;
        y_tst(c:c+length(eval(sprintf('imgs(%d).%s',j, field)))-1,:) = eval(sprintf('imgs(%d).%s',j, field));
        I_tst(c:c+length(eval(sprintf('imgs(%d).%s',j, field)))-1,:) = ...
            ones(length(eval(sprintf('imgs(%d).%s',j, field))),1)*j;
        c = c + length(eval(sprintf('imgs(%d).%s',j, field)));
    end
    hdf5write(fullfile(out_dir, 'test.hdf5'), ...
        'I_tst', I_tst, ...
        'X_tst', X_tst, ...
        'y_tst', y_tst);

    %----------------------------------------------------------------------
    % construct training
    %----------------------------------------------------------------------
    c = 0;
    for i=1:length(idx_trn)
        j = idx_trn(i);
        c = c + length(eval(sprintf('imgs(%d).%s',j, field)));
    end
    X_trn = zeros(c, 4096);
    y_trn = zeros(c, 1);
    c = 1;
    for i=1:length(idx_trn)
        j = idx_trn(i);
        X_trn(c:c+length(eval(sprintf('imgs(%d).%s',j, field)))-1,:) = imgs(j).feat;
        y_trn(c:c+length(eval(sprintf('imgs(%d).%s',j, field)))-1,:) = eval(sprintf('imgs(%d).%s',j, field));
        c = c + length(eval(sprintf('imgs(%d).%s',j, field)));
    end
    fprintf('[%s]: extracted %d activations (training)\n', ...
        datestr(datetime('now')), length(y_trn));
    hdf5write(fullfile(out_dir, 'train.hdf5'), ...
        'X_trn', X_trn, ...
        'y_trn', y_trn);


    %----------------------------------------------------------------------
    % construct validation data
    %----------------------------------------------------------------------
    c = 0;
    for i=1:length(idx_val)
        j = idx_val(i);
        c = c + length(eval(sprintf('imgs(%d).%s',j, field)));
    end
    X_val = zeros(c, 4096);
    y_val = zeros(c, 1);
    c = 1;
    for i=1:length(idx_val)
        j = idx_val(i);
        X_val(c:c+length(eval(sprintf('imgs(%d).%s',j, field)))-1,:) = imgs(j).feat;
        y_val(c:c+length(eval(sprintf('imgs(%d).%s',j, field)))-1,:) = eval(sprintf('imgs(%d).%s',j, field));
        c = c + length(eval(sprintf('imgs(%d).%s',j, field)));
    end
    fprintf('[%s]: extracted %d activations (validation)\n', ...
        datestr(datetime('now')), length(y_val));
    hdf5write(fullfile(out_dir, 'validation.hdf5'), ...
        'X_val', X_val, ...
        'y_val', y_val);

    %----------------------------------------------------------------------
    % store data in struct and in HDF5 files to HDD
    %----------------------------------------------------------------------
    data.X_trn = X_trn;
    data.y_trn = y_trn;
    data.X_val = X_val;
    data.y_val = y_val;
    data.X_tst = X_tst;
    data.y_tst = y_tst;
    data.I_tst = I_tst;

    data.idx_ext = idx_ext; % index into 'imgs'
    data.idx_tst = idx_tst; % index into 'imgs'
    data.idx_trn = idx_trn; % index into 'idx_ext'
    data.idx_val = idx_val; % index into 'idx_ext'

end