function generate_COR_data( config, DataMatrix, DataMatrix_img2idx, selection, out_file )

debug = 1;

OBJ_SCORE_START   = 4100;  % column in DataMatrix where detection scores start
FC7_FEATURES      = 1:4096;% columns of FC7 features
OBJ_DEPTH         = 4098;  % column where depth information is stored
OBJ_ANGLE         = 4099;  % column where angle information is stored

load( 'object_classes' );

Nobjects = length( object_classes ); %#ok<USENS>

% Intersect indices of all images with selection -> only the images that
% are in both sets remain. 
[idx, ~, ~] = intersect( DataMatrix_img2idx(:,1), selection );

use = [];
for j=1:length( idx )
    
    % Get interval of indices into DataMatrix where the features for the
    % image with index idx( j ) reside.
    idx_beg = DataMatrix_img2idx( idx( j ), 2 );
    idx_end = idx_beg + DataMatrix_img2idx( idx( j ), 3 ) - 1;
    
    r = idx_beg:idx_end;
    use = [use; r(:)]; %#ok<AGROW>
    
end

DataMatrix = DataMatrix(use,:);

for i=1:Nobjects
    
    if strcmp( object_classes{ i }, '__background__' )
        continue;
    end
    if strcmp( object_classes{ i }, 'others' )
        continue;
    end
    
    object = object_classes{ i };
    
    object_dir = fullfile( config.SUNRGBD_common, 'objects', object );
    
    if ~exist( object_dir, 'dir' )
        mkdir( object_dir );
    end
    
    object_score = DataMatrix(:, OBJ_SCORE_START + i - 1);
    
    pos =  object_score > 0.5 ;
    
    object_data = DataMatrix( pos, :);
    
    object_X = object_data(:, FC7_FEATURES);
    object_Y = object_data(:, [OBJ_DEPTH OBJ_ANGLE]);
    
    object_X_file = fullfile( object_dir, out_file );
    
    hdf5write( object_X_file, 'X', object_X, 'Y', object_Y );    
    
    if debug
        
        fprintf('%.5d x %.5d | %s\n', ...
            size( object_X, 1 ), ...
            size( object_X, 2 ), object);
        
    end
    
    
    
end
