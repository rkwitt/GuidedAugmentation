function [info, stat, keys] = SUNRGBD_stats( data )

load( 'object_classes' ); % object classes for SUNRGBD object detector

Nimages = length( data );

NO_VAL = -1000;

info = containers.Map;

for i=1:Nimages
     
    disp( i );
    
    scores = data(i).CNN_scores;
    scores(:,1:2) = 0; % __background__ + others
    
    pos = find( data(i).attr_depth ~= NO_VAL );
    
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
        
        tmp_info.angles = [tmp_info.angles data(i).attr_angle( pos( m(k) ) )];
        tmp_info.depths = [tmp_info.depths data(i).attr_depth( pos( m(k) ) )];
        tmp_info.num = tmp_info.num + 1;
        info(object) = tmp_info;
        
    end
    
end

keys = info.keys;

stat = zeros( length( keys ), 5 );

for i=1:length( keys )
   
    key = keys{i};
    stat(i,1) = info( key ).num;
    stat(i,2) = min( info( key ).depths );
    stat(i,3) = max( info( key ).depths );
    stat(i,4) = min( info( key ).angles );
    stat(i,5) = max( info( key ).angles );

end
    
[~, si] = sort( stat,1 );
for i=1:length( si )
    
  fprintf('%.5d | min=%.5f, max=%.5f | min=%.5f, max=%.5f | %s\n', ...
      stat( si(i), 1), ...
      stat( si(i), 2), ...
      stat( si(i), 3), ...
      stat( si(i), 4), ...
      stat( si(i), 5), ...
      keys{ si(i) });
    
end





