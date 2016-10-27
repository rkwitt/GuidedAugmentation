function [info, stat] = SUNRGBD_stats( MetaData_withFeatures, selection )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute statistics of detections per object and min/max values for the
% attributes depth + pose;
%
% selection ... indices for images to be used for statistics computation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


debug = 1;

load( 'object_classes' ); % object classes for SUNRGBD object detector

Nimages = length( selection );

NO_VAL = -1000;

info = containers.Map;

for i=1:Nimages
     
    disp(i);
    idx = selection( i ); % image index
    
    % get CNN scores for image selection( i )
    scores = MetaData_withFeatures( idx ).CNN_scores;
    
    scores(:,1:2) = 0; % __background__ + others
    
    % discard features where we have no depth/pose
    pos = find( MetaData_withFeatures( idx ).attr_depth ~= NO_VAL );
    
    [m,n] = find(scores>0.5);
    
    % m ... index of the m-th detected object
    % n ... index of the n-th detected object class
    for k=1:length(m)
        
        object = object_classes{ n(k) }; %#ok<USENS>
       
        if ~isKey( info, object )
            
            prop.num = 1;
            prop.depths = [];
            prop.angles = [];
            info( object ) = prop;
            
        end
    
        tmp_info = info( object );
        
        tmp_info.angles = [tmp_info.angles MetaData_withFeatures( idx ).attr_angle( pos( m(k) ) )];
        tmp_info.depths = [tmp_info.depths MetaData_withFeatures( idx ).attr_depth( pos( m(k) ) )];
        tmp_info.num = tmp_info.num + 1;
        info(object) = tmp_info;
        
    end
    
end

keys = info.keys;

assert( length( keys ) == length( object_classes) - 2 );

stat = zeros( length( keys ), 5 );

cnt = 1;
for i=3:length( object_classes )

    key = object_classes{i};
    stat(cnt,1) = info( key ).num;
    stat(cnt,2) = min( info( key ).depths );
    stat(cnt,3) = max( info( key ).depths );
    stat(cnt,4) = min( info( key ).angles );
    stat(cnt,5) = max( info( key ).angles );
    
    cnt = cnt + 1;

end

if debug 

    for i=1:size(stat,1);

      fprintf('%.5d | min=%.5f, max=%.5f | min=%.5f, max=%.5f | %s\n', ...
          stat( i, 1), ...
          stat( i, 2), ...
          stat( i, 3), ...
          stat( i, 4), ...
          stat( i, 5), ...
          object_classes{ i+2 });

    end

end




