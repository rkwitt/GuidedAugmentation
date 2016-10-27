function [info,stat,keys] = overlap_statistics( MetaData )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Object statistics for Selective Search bounding boxes overlapping with
% the ground-truth bounding boxes.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NO_VAL = -1000;

Nimages = length( MetaData );

info = containers.Map;

for i=1:Nimages
    
    depths = MetaData(i).attr_depth;
    angles = MetaData(i).attr_angle;
    labels = MetaData(i).prop_label;
    
    pos_depths = find( depths ~= NO_VAL );
    pos_angles = find( angles ~= NO_VAL );
    assert( length( pos_depths ) == length( pos_angles ) );
    
    pos = pos_depths;
    
    for c=1:length( pos );
    
        object = labels{ pos(c) };
        
        if ~isKey( info, object )
            
            prop.depths = [];
            prop.angles = [];
            info(object) = prop;
            
        end
        
        tmp_info = info(object);
        
        tmp_info.angles = [tmp_info.angles; angles( pos(c) )];
        tmp_info.depths = [tmp_info.depths; depths( pos(c) )];
        info(object) = tmp_info;

    end
    
end

keys = info.keys;

stat = zeros( length( keys ), 7 );

for i=1:length( keys )
   
    key = keys{i};
    stat(i, 1) = length( info(key).depths ); % nr. of objects found
    
    stat(i, 2) = range( info(key).depths );  % depth range
    stat(i, 3) =   min( info(key).depths );  % min depth
    stat(i, 4) =   max( info(key).depths );  % max depth
    
    stat(i, 5) = range( info(key).angles );  % angle range
    stat(i, 6) =   min( info(key).angles );  % min angle
    stat(i, 7) =   max( info(key).angles );  % max angle

end







