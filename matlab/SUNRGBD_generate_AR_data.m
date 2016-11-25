function SUNRGBD_generate_AR_data( config, DataMatrix, DataMatrix_img2idx, selection, prefix, out_file )

OBJ_SCORE_START   = 4100;  % column in DataMatrix where detection scores start
FC7_FEATURES      = 1:4096;% columns of FC7 features
OBJ_DEPTH         = 4098;  % column where depth information is stored
OBJ_ANGLE         = 4099;  % column where angle information is stored

load( 'matfiles/SUNRGBD_objects' );
Nobjects = length( object_classes ); %#ok<USENS>

% Intersect indices of all images with selection 
% -> only the images that are in both sets remain. 
[idx, ~, ~] = intersect( DataMatrix_img2idx(:,1), selection );

use = [];
for j=1:length( idx )
    
    % Get interval of indices into DataMatrix 
    % where the features for the image with 
    % index idx( j ) resides.
    idx_beg = DataMatrix_img2idx( idx( j ), 2 );
    idx_end = idx_beg + DataMatrix_img2idx( idx( j ), 3 ) - 1;
    
    r = idx_beg:idx_end;
    use = [use; r(:)]; %#ok<AGROW>
    
end

% Restrict data matrix to those features which
% correspond to images in our selection.
DataMatrix = DataMatrix(use,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate data for object-agnostic AR training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

object_score = DataMatrix(:, ...
    OBJ_SCORE_START: OBJ_SCORE_START + length(object_classes) - 1);

% skip __background__ and others and then
% get all detections with scores > 0.5
pos =  object_score(:,3:end) > 0.5 ;

assert( sum(sum(pos,2)<=1) == length(pos));

pos = logical(sum(pos,2));

% get only those entries of the DataMatix with
% object scores > 0.5 (except for background and
% others.
object_data = DataMatrix( pos, :);

% object_X ... CNN features
% object_Y ... Attribute values for DEPTH and POSE
% object_L ... CNN scores for features with scores > 0.5
object_X = object_data(:, FC7_FEATURES);
object_Y = object_data(:, [OBJ_DEPTH OBJ_ANGLE]);
object_L = object_data(:, OBJ_SCORE_START:OBJ_SCORE_START+Nobjects-1);

X = [];
Y = [];
for m=3:21
    n = length(find(object_L(:,m)>0.5));
    if n < 100
        fprintf('skipping %d\n', m);
        continue;
    end
    pp = randsample(find(object_L(:,m)>0.5), 100);
    X = [X; object_X(pp,:)]; %#ok<AGROW>
    Y = [Y; object_Y(pp,:)]; %#ok<AGROW>
end

object_dir = fullfile( config.outdir, prefix );
object_X_file = fullfile( object_dir, out_file );
hdf5write( object_X_file, 'X', X, 'Y', Y );

clear object_score
clear object_X
clear object_Y
clear pos


for i=1:Nobjects
    
    if strcmp( object_classes{ i }, '__background__' )
        continue;
    end
    if strcmp( object_classes{ i }, 'others' )
        continue;
    end
    
    object = object_classes{ i };
    
    object_dir = fullfile( config.outdir, prefix, object );
    
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
