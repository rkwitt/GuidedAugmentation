function generate_covariate_regression_data( DataMatrix, config, img2idx, target_idx, out_file )

OBJ_SCORE_START   = 4100; % column in DataMatrix where detection scores start
OBJ_FC7_BEG       = 1;    % column where FC7 feature starts
OBJ_FC7_END       = 4096; % column where FC7 feature ends
OBJ_DEPTH         = 4098; % column where depth information is stored
OBJ_ANGLE         = 4099; % column where angle information is stored

load( 'object_classes' ); % object classes for SUNRGBD object detector

Nobjects = length( object_classes ); %#ok<USENS>

% since the first column of img2idx holds the image number, we can simply
% intersect with the target_idx (e.g., training indices) and only get those
% indices that we actually want. Columns 2-3 then hold the position where
% the actual features for a particular image start and how many of them we
% have.
[idx, ~, ~] = intersect( img2idx(:,1), target_idx );

use = [];
for j=1:length(idx)
    
    r = img2idx( idx( j ), 2 ):img2idx( idx( j ), 2 ) + img2idx( idx( j ), 3 ) - 1;
    use = [use; r(:)]; %#ok<AGROW>
    
end

% prune DataMatrix to only those entries that are relevant for the desired
% images, specified via target_idx.
Y = DataMatrix(use,:);

for i=3:Nobjects % ignore __background__ + others
   
    object = object_classes{i};
    disp( object );
    
    object_dir = fullfile( config.SUNRGBD_common, object );
    
    if ~exist( object_dir, 'dir' )
        mkdir( object_dir );
    end
    
    object_idx = OBJ_SCORE_START + i - 1;
    
    object_score = Y(:,object_idx);
    
    pos = find( object_score > 0.5 );
    
    object_X = Y(pos, OBJ_FC7_BEG:OBJ_FC7_END);
    object_Y = Y(pos, [OBJ_DEPTH OBJ_ANGLE]);
    
    object_X_file = fullfile( object_dir, out_file );
        
    hdf5write( object_X_file, 'X', object_X, 'Y', object_Y );    
    
end
