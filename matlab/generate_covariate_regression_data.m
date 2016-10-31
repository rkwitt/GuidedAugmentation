function generate_covariate_regression_data( config, DataMatrix, DataMatrix_img2idx, selection, out_file )

debug = 1;

OBJ_SCORE_START   = 4100; % column in DataMatrix where detection scores start
OBJ_FC7_BEG       = 1;    % column where FC7 feature starts
OBJ_FC7_END       = 4096; % column where FC7 feature ends
OBJ_DEPTH         = 4098; % column where depth information is stored
OBJ_ANGLE         = 4099; % column where angle information is stored

load( 'object_classes' ); % object classes for SUNRGBD object detector

Nobjects = length( object_classes ); %#ok<USENS>

[idx, ~, ~] = intersect( DataMatrix_img2idx(:,1), selection );

use = [];
for j=1:length( idx )
    
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
    
    object_X = object_data(:, OBJ_FC7_BEG:OBJ_FC7_END );
    object_Y = object_data(:, [OBJ_DEPTH OBJ_ANGLE]);
    
    object_X_file = fullfile( object_dir, out_file );
        
    hdf5write( object_X_file, 'X_trn', object_X, 'Y_trn', object_Y );    
    
    if debug
        
        fprintf('%.5d x %.5d | %s\n', ...
            size( object_X, 1 ), ...
            size( object_X, 2 ), object);
        
    end
    
    
    
end
