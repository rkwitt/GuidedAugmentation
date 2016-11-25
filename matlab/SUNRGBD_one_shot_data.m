function SUNRGBD_one_shot_data( config, SUNRGBDMeta_correct2D, os_classes, out_dir, selection, debug )

    if ~exist( out_dir, 'dir' )
        mkdir( out_dir );
    end

    fid = fopen(fullfile( out_dir, 'allimgs.txt'), 'w');
       
    cnt = 1;
    for i=1:length( selection )

        %
        % get image ID
        %
        idx = selection( i );
        
        %
        % get information for object in 3D BB
        %
        meta = SUNRGBDMeta_correct2D( idx ).groundtruth3DBB;
        
        %
        % get the number of annotated objects 
        %
        Ndet = length( meta );
    
        %
        % make sure we have an equal number of 2D bounding boxes
        %
        assert( Ndet == length(SUNRGBDMeta_correct2D( idx ).groundtruth2DBB) );

        boxes = [];
        labels = [];
        
        %
        % read image
        %
        img_file = fullfile( ...
            config.SUNRGBD_dir, ...
            SUNRGBDMeta_correct2D( idx ).sequenceName, ...
            'image', ...
            SUNRGBDMeta_correct2D( idx ).rgbname);
        
        if ~exist( img_file, 'file' )
            fprintf('File %d not found!\n', img_file);
        end
        
        %
        % iterate over objects and extract the ones that we want 
        %
        for j=1:Ndet

            label = meta(j).classname;
            
            for c=1:length( os_classes )
            
                %
                % take object if it is in our list of object classes that
                % we want.
                %
                if strcmp( label, os_classes{c} )

                    bb = SUNRGBDMeta_correct2D( idx ).groundtruth2DBB(j).gtBb2D;
                    
                    if isempty( bb )
                        continue;
                    end
                    
                    %
                    % draw BB in debug mode
                    %
                    
                    if debug
                        
                        im = imread( img_file );
                        imshow( im );
                        rectangle( 'Position', ...
                            [bb(1), bb(2), bb(3), bb(4)], ...
                            'EdgeColor','blue', 'LineWidth',3);
                        title( os_classes{c} );
                        pause;
                        close all;
                        
                    end
                    
                    %
                    % store BB as [x1,y1,x2,y2]; originally BBs are stored
                    % as [x,y,w,h] in the SUNRGBD mat file
                    %                 
                    tmp_boxes = round( [bb(1), bb(2), bb(3)+bb(1), bb(4)+bb(2)] );
                    
                    %
                    % add box and assign a label ID, i.e., the ID that
                    % specifies the object class.
                    %
                    boxes = [boxes; tmp_boxes]; %#ok<AGROW>
                    labels = [labels; c]; %#ok<AGROW>
                    
                end
                
            end
            
        end
            
        %
        % in case we did not find anything in the i-th image, skip ...
        %
        if isempty( boxes )
            continue;
        end
        
        out_file = sprintf( 'image_%.5d.jpg', cnt );
        out_boxes = sprintf( 'image_%.5d_ss_boxes.mat', cnt );
        out_labels =  sprintf( 'image_%.5d_ss_labels.mat', cnt );
        
        copyfile( img_file , fullfile( out_dir, out_file ) );
        save( fullfile( out_dir, out_boxes ), 'boxes' );
        save( fullfile( out_dir, out_labels ), 'labels' );
        
        fprintf(fid, 'image_%.5d\n', cnt);
        
        fprintf('[%d,%d]: %s\n', size(boxes,1), cnt, out_file);
        cnt = cnt + 1;
        
    end
    
    fclose(fid);