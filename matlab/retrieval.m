function vals = retrieval(obj_mat, mat_field, data, activation_file)

    tmp = load(obj_mat);
    imgs = eval(sprintf('tmp.%s', mat_field));
    activations = hdf5read(activation_file, 'predicted_activations')';
  
    vals = zeros(200,3);
    p = randsample(size(activations,1),200);
    for i=1:length(p)    
        %D = pdist2(data.X_tst(p(i),:), data.X_tst);
        %[~, idx] = sort(D,2);
        %vals(i,:) = [data.y_tst(p(i)) mean(data.y_tst(idx(2:10)))];
        
        D = pdist2(activations(p(i),:), data.X_tst);
        [~, idx] = sort(D,2);
        vals(i,:) = data.y_tst(idx(1:3))';
        
        %vals(i,:) = [data.y_tst(p(i)) mean(data.y_tst(idx(2:10)))];
        
        
    end
    
    
    
    
end