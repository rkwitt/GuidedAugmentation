function info = gather_one_shot_data( base )

img_files = dir( fullfile( base, '*.jpg' ) );

Nimages = length( img_files );

EDN_FILE_FMT = '%s_bbox_features_OAEDN.mat';
ORG_FILE_FMT = '%s_bbox_features.mat';
LAB_FILE_FMT = '%s_ss_labels.mat';

info = containers.Map();

for i=1:Nimages
    
    disp(i);
    [~,img_base,~] = fileparts( img_files(i).name );
    
    % load necessary files
    edn_file = fullfile( base, sprintf( EDN_FILE_FMT, img_base ) );
    org_file = fullfile( base, sprintf( ORG_FILE_FMT, img_base ) );
    lab_file = fullfile( base, sprintf( LAB_FILE_FMT, img_base ) );
        
    assert( exist( edn_file, 'file' ) > 0 );
    assert( exist( org_file, 'file' ) > 0 );
    assert( exist( lab_file, 'file' ) > 0 );
   
    %
    % load original features and object labels to those features
    %    
    load(org_file);
    load(lab_file);
    Xorg = CNN_feature;
    
    
    %
    % load EDN-generated features 
    %
    load(edn_file);        
    Xedn = CNN_feature;
    
    clear CNN_feature;
    clear CNN_scores;
    
   
    label_ids = unique(labels, 'stable'); % ensure order is the same as order of occurrence
     
    for l=1:length(label_ids)
        
        %
        % convert label num -> string (used as key later)
        %
        label_str = num2str(label_ids(l));
        if ~info.isKey(label_str)
            info(label_str) = [];
        end

        %
        % find ALL positions where, in original label file, label_ids(l)
        % occurs. 
        %
        pos = find(labels==label_ids(l));
        
        %
        % get information structure for label_ids(l)
        %
        O = info(label_str);
        
        %
        % iterate over positions where label_ids(l) occurs, i.e., one ID
        % equals one feature vector.
        %
        for j=1:length(pos)
    
            Xorg_obj = Xorg(pos(j), :);
            
            %
            % the augmented features have associated information that
            % indicates to which object, in the order as they occur in the
            % original feature file, the augmented features belong to
            % (starting from 0).
            % 
            idx = CNN_metadata(:,end) == pos(j)-1; %#ok<NODEF>
            
            Xedn_obj = Xedn(idx,:);
            
            assert(unique(CNN_metadata(idx,1)) == label_ids(l) );
           
            O = [O; {Xorg_obj, Xedn_obj}];
                        
        end
        
        info(label_str) = O;
        
    end
        
end

%close(h);









