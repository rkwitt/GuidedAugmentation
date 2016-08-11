function object_info = ga_extract_original_object_BB(SUNRGBDMeta, query_obj, verbose)

    % N images
    N = length(SUNRGBDMeta);
    
    s=1;
    for i=1:N
        
        if mod(i,100)==0
            fprintf('Status: Image %.4d\n', i);
        end
        
        data = SUNRGBDMeta(i);
        
        nobjs = size(data.groundtruth3DBB, 2);
        for k = 1:nobjs
            
            if(strcmp(data.groundtruth3DBB(k).classname, query_obj))
                
                object_info(s).img_idx = i;
                object_info(s).centroid = data.groundtruth3DBB(k).centroid;
                object_info(s).depth = norm(object_info(s).centroid);
                object_info(s).bb = data.groundtruth2DBB(k).gtBb2D;
                object_info(s).img_file = fullfile(getenv('SUNRGBD_dir'), data.sequenceName, 'image', data.rgbname);
                
                if verbose
                    im = imread(object_info(s).img_file);
                    imshow(im);
                    hold;
                    rectangle('Position',[...
                        object_info(s).bb(1), ...
                        object_info(s).bb(2), ...
                        object_info(s).bb(3), ...
                        object_info(s).bb(4)], 'EdgeColor', 'blue', 'LineWidth',3);
                    pause;
                    close all;
                    
                end
                
                s = s+1;
            end
        end
        
    end
end