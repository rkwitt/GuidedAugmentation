function obj_info = ga_extract_refined_object_BB(SUNRGBDMeta, query_obj, verbose)

    % N images
    N = length(SUNRGBDMeta);
    
    obj_cnt = 0;
    
    for i=1:N
        
        if mod(i,100)==0
            fprintf('Status: Image %.4d\n', i);
        end
        
        data = SUNRGBDMeta(i);
        
        seg_mat_file = fullfile(...
            getenv('SUNRGBD_dir'), ...
            SUNRGBDMeta(i).sequenceName, ...
            'seg.mat');
        
        if ~exist(seg_mat_file, 'file')
            fprintf('Missing %s -> skipping ...\n', ...
                seg_mat_file);
            continue;
        end
        
        seg = load(seg_mat_file, ...
            '-mat', ...
            'seglabel', ...
            'names');
        
        seg_label = seg.seglabel;
        seg_names = seg.names;
        
        for j=1:length(seg_names)
            
            target = seg_names{j};
            if strcmp(target, query_obj)
      
                obj_seg = seg_label == j;
                rp = regionprops(obj_seg, 'BoundingBox' );                
                if isempty(rp)
                    continue;
                end
                bb = rp.BoundingBox;
                
                
                found_cnt = 0;
                found_box = 0; 
                nobjs = size(data.groundtruth3DBB, 2);
                for k = 1:nobjs
                    if(strcmp(data.groundtruth3DBB(k).classname, query_obj))
                        
                        bb_tmp = data.groundtruth2DBB(k).gtBb2D;
                        IoU = bboxOverlapRatio(bb_tmp, bb);
                        if (IoU >=0.9)
                            found_cnt = found_cnt + 1;
                            found_box = bb_tmp;
                            found_dep = norm(data.groundtruth3DBB(k).centroid);
                        end
                    end
                end
                    
                if ~found_cnt
                    continue;
                end
                assert(found_cnt==1);
                obj_cnt = obj_cnt + 1;
                
                rgb_img_file = fullfile(...
                    getenv('SUNRGBD_dir'), ...
                    SUNRGBDMeta(i).sequenceName, ...
                    'image', ...
                    SUNRGBDMeta(i).rgbname);
                
                obj_info(obj_cnt).img_idx = i;                  % image index
                obj_info(obj_cnt).depth = found_dep;            % obj. depth
                obj_info(obj_cnt).bb = bb;                      % bbox
                obj_info(obj_cnt).seg_file = seg_mat_file;      % segmentation file
                obj_info(obj_cnt).img_file = rgb_img_file;       % image file
                
                if verbose
                    disp(obj_cnt);
                end
                
                if verbose
                    subplot(1,3,1);
                    imshow(seg_label,[]);
                end

                if verbose
                    subplot(1,3,2);
                    imshow(obj_seg,[]);
                end
                
                if verbose
                    subplot(1,3,3);
                    im = imread(rgb_img_file);
                    imshow(im);
                    hold;
                    rectangle('Position', ...
                        [bb(1),bb(2),bb(3),bb(4) ], ...
                        'EdgeColor','blue', 'LineWidth',3);

                    rectangle('Position', ...
                        [found_box(1),found_box(2),found_box(3),found_box(4) ], ...
                        'EdgeColor','red', 'LineWidth',3);
                    pause;
                end
                
            end
            
        end
        
    end
    
end