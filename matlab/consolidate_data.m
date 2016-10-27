function DataMatrix = consolidate_data( MetaData_withFeatures, config )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Consolidate all data into one huge matrix for easier processing and
% extraction of subsets of the data.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

debug = 1;

NO_VAL = -1000;

load( 'object_classes' ); % object classes for SUNRGBD object detector

Nobjects = length( object_classes );

Nimages = length( MetaData_withFeatures );

% compute the total number of detections over all images
Ndetections = 0;
for i=1:Nimages
   
    Ndetections = Ndetections + size( MetaData_withFeatures(i).CNN_features, 1 );
    
end

if debug 
    disp('allocating space ...');
end
    
DataMatrix = zeros( Ndetections, 4096 + 1 + 2 + length( object_classes ) );
DataMatrix = single( DataMatrix );
DataMatrix_img2idx = zeros( Nimages, 3 );
 
cnt = 1;
for i=1:Nimages
   
   disp(i);
   N = size( MetaData_withFeatures(i).CNN_features, 1 );
   
   depths = MetaData_withFeatures(i).attr_depth;
   angles = MetaData_withFeatures(i).attr_angle;
   
   pos_depths = find( depths ~= NO_VAL );
   pos_angles = find( angles ~= NO_VAL );
   
   % sanity check(s)
   assert( sum( pos_depths == pos_angles ) == length( pos_depths ) );
   assert( length( pos_depths ) == N );
   
   pos = pos_depths;
   
   DataMatrix(cnt:cnt+N-1, 1:4096)               = MetaData_withFeatures(i).CNN_features; % FC7
   DataMatrix(cnt:cnt+N-1, 4097)                 = ones(N,1)*i;                           % Image idx
   DataMatrix(cnt:cnt+N-1, 4098)                 = depths( pos )';                        % Depths
   DataMatrix(cnt:cnt+N-1, 4099)                 = angles( pos )';                        % Angles
   DataMatrix(cnt:cnt+N-1, 4100:4100+Nobjects-1) = MetaData_withFeatures(i).CNN_scores;   % Scores
   
   DataMatrix_img2idx(i,1:3) = [i cnt N];
   
   cnt = cnt + N;
    
end

DataMatrix = single( DataMatrix );
save( fullfile( config.SUNRGBD_common, 'DataMatrix.mat' ), 'DataMatrix', '-v7.3' );
save( fullfile( config.SUNRGBD_common, 'DataMatrix_img2idx.mat' ), 'img2idx', '-v7.3');






