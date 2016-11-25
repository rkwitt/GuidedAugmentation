function stat = SUNRGBD_stats( DataMatrix, DataMatrix_img2idx, selection )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute statistics of detections per object and min/max values for the
% attributes depth + pose;
%
% selection ... indices for images to be used for statistics computation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


OBJ_SCORE_START = 4100;
OBJ_DEPTH = 4098;
OBJ_ANGLE = 4099;

debug = 1;

load( 'object_classes' ); % object classes for SUNRGBD object detector

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

stat = zeros( Nobjects, 5 );

for k=1:Nobjects
   
    disp( object_classes{ k } ); 
   
    object_data = DataMatrix(use,:);
    
    object_scores = object_data(:, OBJ_SCORE_START + k - 1);
    
    pos =  object_scores > 0.5 ;
   
    object_data = object_data( pos, :);
    
    object_depth = object_data(:, OBJ_DEPTH);
    object_angle = object_data(:, OBJ_ANGLE);
    
    stat(k, 1) = size( object_data, 1 );
    stat(k, 2) =  min( object_depth );
    stat(k, 3) =  max( object_depth );
    stat(k, 4) =  rad2deg(min( object_angle ));
    stat(k, 5) =  rad2deg(max( object_angle ));
    
end    
    
   
if debug 

    for i=1:size(stat,1);

      fprintf('%.5d | min=%.5f, max=%.5f | min=%.5f, max=%.5f | %s\n', ...
          stat( i, 1), ...
          stat( i, 2), ...
          stat( i, 3), ...
          stat( i, 4), ...
          stat( i, 5), ...
          object_classes{ i });

    end

end
