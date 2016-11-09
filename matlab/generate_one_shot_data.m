function generate_one_shot_data( config, data, os_classes, out_dir, selection )


    if ~exist( out_dir, 'dir' )
        mkdir( out_dir );
    end
    
    cnt = 1;
    for i=1:length( selection )

        idx = selection( i );
        
        meta = data( idx ).groundtruth3DBB;
        
        Ndet = length( meta );
    
        assert( Ndet == length(data( idx ).groundtruth2DBB) );

        boxes = [];
        labels = [];
        
        img_file = fullfile( ...
            config.SUNRGBD_common, ...
            data( idx ).sequenceName, ...
            'image', ...
            data( idx ).rgbname);
        
        if ~exist( img_file, 'file' )
            fprintf('File %d not found!\n', img_file);
        end
        
        for j=1:Ndet

            label = meta(j).classname;
            
            for c=1:length( os_classes )
            
                if strcmp( label, os_classes{c} )

                    bb = data( idx ).groundtruth2DBB(j).gtBb2D;
                    
                    if isempty( bb )
                        continue;
                    end
                    
%                     im = imread( img_file );                    
%                     imshow( im );
%                     rectangle( 'Position', ...
%                         [bb(1), bb(2), bb(3), bb(4)], ...
%                         'EdgeColor','blue', 'LineWidth',3);
%                     title( os_classes{c} );
                    
                    tmp_boxes = round( [bb(1), bb(2), bb(3)+bb(1), bb(4)+bb(2)] ); 
                    boxes = [boxes; tmp_boxes];
                    labels = [labels; c];
                    
                end
                
            end
            
        end
            
        if isempty( boxes )
            continue;
        end
        
        out_file = sprintf( 'image_%.5d.jpg', cnt );
        out_boxes = sprintf( 'image_%.5d_ss_boxes.mat', cnt );
        out_labels =  sprintf( 'image_%.5d_ss_labels.mat', cnt );
        
        copyfile( img_file , fullfile( out_dir, out_file ) );
        save( fullfile( out_dir, out_boxes ), 'boxes' );
        save( fullfile( out_dir, out_labels ), 'labels' );
        
        fprintf('[%d,%d]: %s\n', size(boxes,1), cnt, out_file);
        cnt = cnt + 1;
        
    end