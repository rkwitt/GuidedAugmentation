function generate_feature_regression_data( DataMatrix, DataMatrix_img2idx, selection, gamma )

OBJ_SCORE_START   = 4100; 
OBJ_DEPTH         = 4098; 
FC7_FEATURES      = 1:4096; 

load( 'object_classes' );

Nobjects = length( object_classes ); %#ok<USENS>

image_indices = DataMatrix_img2idx(:,1);

[idx, ~, ~] = intersect( image_indices, selection );

use = [];
for j=1:length( idx )
    
    idx_beg = DataMatrix_img2idx( idx( j ), 2 );
    idx_end = idx_beg + DataMatrix_img2idx( idx( j ), 3 ) - 1;
    
    r = idx_beg:idx_end;
    use = [use; r(:)]; %#ok<AGROW>
    
end

for k=1:Nobjects
    
    if strcmp( object_classes{ k }, '__background__' )
        continue;
    end
    if strcmp( object_classes{ k }, 'others' )
        continue;
    end
    
    disp( object_classes{ k } );
    
    object_data = DataMatrix(use,:);
    
    object_score = object_data(:, OBJ_SCORE_START + k - 1);
    
    pos =  object_score > 0.5 ;
   
    object_data = object_data( pos, :);
   
    object_X = object_data( :, FC7_FEATURES );
    object_Y = object_data( :, OBJ_DEPTH );
    
    binning_info = binning(object_Y, 5, gamma);
    disp(binning_info)
    
    
end






