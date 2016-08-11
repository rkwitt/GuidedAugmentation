function ga_save_fastRCNN(objs, out_dir, rmPrefix)


% create output dir
if ~exist(out_dir, 'dir')
    mkdir(out_dir)
end

img_list_file = fullfile(out_dir, 'img_list.txt');
img_list_fid = fopen(img_list_file,'w');


N=length(objs);
for i=1:N
   
    tmp = objs(i).bb;
    if isempty(tmp)
        fprintf('Found empty BB @ %.5d.\n', i);
        continue;
    end
    
    boxes = [tmp(1) tmp(2) tmp(1)+tmp(3) tmp(2)+tmp(4)];
    
    
    bb_mat_file = sprintf('bb_%.5d.mat', i);
    bb_mat_full_file = fullfile(out_dir, ...
        bb_mat_file);
    
    img_file = objs(i).img_file(length(rmPrefix)+1:end);
    
    fprintf(img_list_fid, ...
        '%s %s %.3f\n', ...
        img_file, ...
        bb_mat_file, ...
        objs(i).depth);
    
    save(bb_mat_full_file, 'boxes');
    
end

fclose(img_list_fid);


