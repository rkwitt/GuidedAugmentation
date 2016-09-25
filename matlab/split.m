function out = split(data, steps, delta, gamma, out_dir, verbose)

    cnt=1;
    for i=1:length(steps)

        l0 = steps(i);
        l1 = steps(i)+delta;

        y_trn = data.y_trn;
        i_trn = find(y_trn>=l0 & y_trn<l1);
        y_trn = y_trn(i_trn);
        if (length(i_trn)<gamma)
            continue;
        end

        if verbose
            fprintf('[%s]: l0=%.2f, l1=%.2f => %d samples\n', ...
                datestr(datetime('now')), l0, l1, length(y_trn));
        end
            
        X_source = data.X_trn(i_trn, :);
        X_source_file = fullfile(out_dir, sprintf('train_i%d.hdf5', i));
        hdf5write(X_source_file, 'X_source', X_source);    
        
        y_val = data.y_val;
        i_val = find(y_val>=l0 & y_val<l1);
        y_val = y_val(i_val);
        
        X_validation = data.X_val(i_val, :);
        X_validation_file = fullfile(out_dir, sprintf('validation_i%d.hdf5', i));
        hdf5write(X_validation_file, 'X_validation', X_validation);    
        
        out(cnt).l0 = l0; %#ok<*AGROW>
        out(cnt).l1 = l1;
        out(cnt).y_trn = y_trn;
        out(cnt).i_trn = i_trn;
        out(cnt).y_trn = y_val;
        out(cnt).i_trn = i_val;        
        out(cnt).X_source = X_source;
        out(cnt).X_validation = X_validation;
        out(cnt).X_source_file = X_source_file;
        out(cnt).X_validation_file = X_validation_file;
        cnt = cnt + 1;
    
    end

    fprintf('[%s]: wrote %d source data files\n', ...
        datestr(datetime('now')), cnt);
    
end

