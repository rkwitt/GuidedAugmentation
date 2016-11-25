function MetaData_withFeatures = SUNRGBD_collect_features( MetaData, config )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Augment object_meta data by CNN features and their scores w.r.t. to the
% SUNRGBD object categories.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load( 'object_classes' ); % object classes for SUNRGBD object detector

Nimages = length( MetaData );

NO_VAL = -1000;

MetaData_withFeatures = MetaData;

for i=1:Nimages
    
   tic; 
    
   prefix = 'image_%.5d'; 
    
   feature_file = [sprintf(prefix, i) '_bbox_features.mat'];
   
   load( fullfile( config.SUNRGBD_common, feature_file ) );
   
   % get depth and pose data for i-th image
   depths = MetaData_withFeatures(i).attr_depth;
   angles = MetaData_withFeatures(i).attr_angle;
   
   % find positions where we have depth / pose values
   pos_depths = find( depths ~= NO_VAL );
   pos_angles = find( angles ~= NO_VAL );

   % sanity check(s)
   assert( sum(pos_depths == pos_angles) == length( pos_depths ) );
   assert( length( depths ) == size( CNN_feature, 1 ) );
   
   pos = pos_depths;

   % prune to selection where we have depths + pose information; since pos
   % is a sorted list of positions, the CNN features are also in that
   % order.
   CNN_feature  = CNN_feature(pos, :);
   CNN_scores   = CNN_scores(pos, :); 
   
   MetaData_withFeatures(i).CNN_features = CNN_feature;
   MetaData_withFeatures(i).CNN_scores = CNN_scores;
   
   ela = toc;
   fprintf('%.5d/%.5d | %s [took %.3f seconds]\n', ...
       i, Nimages, feature_file, ela);
   
   clear CNN_feature
   clear CNN_scores
   
end

save( fullfile( config.SUNRGBD_common, 'MetaData_withFeatures.mat'), ...
    'MetaData_withFeatures', '-v7.3' );
