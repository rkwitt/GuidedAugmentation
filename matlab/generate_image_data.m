function generate_image_data( config, DataMatrix, DataMatrix_img2idx, selection )

OBJ_SCORE_START   = 4100;  % column where RCNN object scores start
OBJ_DEPTH         = 4098;  % column where depth information is stored
OBJ_ANGLE         = 4099;  % column where angle information is stored

load( 'object_classes' );

image_indices = DataMatrix_img2idx(:,1);

for s=1:length( selection )
    
    disp( selection( s ) );
    
    pos = find( image_indices == selection( s ) );
    
    assert( length( pos ) == 1 ); % there better be only one image with that ID
    
    idx_beg = DataMatrix_img2idx( pos, 2 );
    idx_end = idx_beg + DataMatrix_img2idx( pos, 3 ) - 1;
    use = idx_beg:idx_end;
    
    image_data = DataMatrix( use, : );
    
    object_scores = image_data(:, OBJ_SCORE_START: OBJ_SCORE_START+length( object_classes) - 1);
    
    [u,v] = find(object_scores > 0.5);
    
    p0 = v == 1; % __background__
    p1 = v == 2; % others
    px = logical( p0 + p1 );
    
    v( px ) = []; % col IDs
    u( px ) = []; % row IDs
    
    X = image_data(u, 1:4096);
    Y = image_data(u, [OBJ_DEPTH OBJ_ANGLE]);
    
    if isempty( X ) 
        continue;
    end
    
    image_data_file = fullfile( config.SUNRGBD_common, ...
        sprintf( 'image_%.5d_data.hdf5', selection( s ) ) );
    
    hdf5write(image_data_file, '/X', X, '/objects', v, '/Y', Y );
    
end