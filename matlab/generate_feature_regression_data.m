function generate_feature_regression_data( DataMatrix, img2idx, selection, stat, minN )

OBJ_SCORE_START   = 4100; 
OBJ_DEPTH         = 4098; 
FC7_FEATURES      = 1:4096; 

load( 'object_classes' );

Nobjects = length( object_classes ); %#ok<USENS>

for k=3:Nobjects % skip __background__ + others
   
    if stat(k-2, 1) < minN
        
        disp(['skipping ' object_classes{ k } ' (too few samples)']);
                
        continue;
    end
    
    image_indices = img2idx(:,1);
    
    [idx, ~, ~] = intersect( image_indices, selection );

    use = [];
    for j=1:length( idx )

        idx_beg = img2idx( idx( j ), 2 );
        idx_end = idx_beg + img2idx( idx( j ), 3 ) - 1;
        
        r = idx_beg:idx_end;
        use = [use; r(:)]; %#ok<AGROW>

    end

    object_data = DataMatrix(use,:);
    
    object_score = object_data(:, OBJ_SCORE_START + (k-2) - 1);
    
    pos =  object_score > 0.5 ;
   
    object_data = object_data( pos, :);
   
    object_X = object_data( :, FC7_FEATURES );
    object_Y = object_data( :, OBJ_DEPTH );
    
    
end






