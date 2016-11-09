function [edn_data, edn_meta, org_data, org_meta] = one_shot( base )

img_files = dir( fullfile( base, '*.jpg' ) );

Nimages = length( img_files );

org_data = [];
org_meta = [];

edn_data = [];
edn_meta = [];

for i=1:Nimages
    
    disp(i);
    [~,img_base,~] = fileparts( img_files(i).name );
    
    edn_file = fullfile( base, sprintf( '%s_bbox_features_OAEDN.mat', img_base ) );
    org_file = fullfile( base, sprintf( '%s_bbox_features.mat', img_base ) );
    org_labels = fullfile( base, sprintf( '%s_ss_labels.mat', img_base ) );
    
    assert( exist( edn_file, 'file' ) > 0 );
    assert( exist( org_file, 'file' ) > 0);
    
    load(org_file);
    load(org_labels);
    
    org_data = [org_data; CNN_feature];  %#ok<AGROW>
    tmp_meta = [ones( size( CNN_feature, 1 ), 1 )*i labels(:)];
    org_meta = [org_meta; tmp_meta];
    
    clear CNN_feature
    clear CNN_scores
    
    load(edn_file);
    CNN_metadata(:,end) = CNN_metadata(:,end)+i;
    edn_data = [edn_data; CNN_feature];  %#ok<AGROW>
    edn_meta = [edn_meta; CNN_metadata]; %#ok<AGROW>
    
    clear CNN_feature
    clear CNN_metadata
    
    
end











