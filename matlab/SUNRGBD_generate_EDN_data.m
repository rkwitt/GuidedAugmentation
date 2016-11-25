function SUNRGBD_generate_EDN_data( config, DataMatrix, DataMatrix_img2idx, selection, gamma, prefix, OBJ_ATTR, binfun, seed )
%
% Generate data for EDN training;
%
% OBJ_ATTR         = 4098; pose
% OBJ_ATTR         = 4099; angle
%
% For depth: binfun = @(x) binbystep(x, 1, 0.5)
% For pose:  binfun = @(x) binbystep(x, deg2rad(45), deg2rad(25))

rng( seed );

OBJ_SCORE_START   = 4100; 
FC7_FEATURES      = 1:4096; 

load( 'object_classes' );

Nobjects = length( object_classes );  %#ok<USENS>

image_indices = DataMatrix_img2idx(:,1);

[idx, ~, ~] = intersect( image_indices, selection );

use = [];
for j=1:length( idx )
    
    idx_beg = DataMatrix_img2idx( idx( j ), 2 );
    idx_end = idx_beg + DataMatrix_img2idx( idx( j ), 3 ) - 1;
    
    r = idx_beg:idx_end;
    use = [use; r(:)]; %#ok<AGROW>
    
end


fid = fopen( fullfile( config.outdir,  ...
    prefix, 'info_agnostic_training.txt' ), 'w' );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate data for object agnostic EDN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

object_data = DataMatrix(use,:);

object_score = object_data(:, OBJ_SCORE_START: OBJ_SCORE_START + length(object_classes) - 1);

pos =  object_score(:,3:end) > 0.5 ;

assert( sum(sum(pos,2)<=1 ) == length(pos));

pos = logical(sum(pos,2));

object_data = DataMatrix( pos, :);

object_X = object_data( :, FC7_FEATURES );
object_Y = object_data( :, OBJ_ATTR ); 
object_L = object_data( :, OBJ_SCORE_START:OBJ_SCORE_START+21-1 );
    
binning_info = binfun( object_Y );
binning_info(binning_info(:,3) < gamma,:) = [];

for bin_j=1:size( binning_info, 1 )
    
    lo = binning_info( bin_j, 1 );
    hi = binning_info( bin_j, 2 );
    
    pos_bin_j =  object_Y >= lo & object_Y < hi ;
    
    object_X_bin_j = object_X( pos_bin_j, : ); % FC7
    object_Y_bin_j = object_Y( pos_bin_j, : ); % Attribute
    object_L_bin_j = object_L( pos_bin_j, : ); % Scores
    
    X = [];
    Y = [];
    for m=3:21
        n = length(find(object_L_bin_j(:,m)>0.5));
        if n < 100
            fprintf('skipping %d\n', m);
            continue;
        end
        pp = randsample(find(object_L_bin_j(:,m)>0.5), 100);
        X = [X; object_X_bin_j(pp,:)];
        Y = [Y; object_Y_bin_j(pp,:)];
    end
    
    object_trn_bin_j = sprintf( 'trn_i%.4d.hdf5', bin_j );
    
    ED_meta_line = strcat(num2str( lo ), ',', ...
        num2str( hi ), ',', object_trn_bin_j );
    
    disp(size(X));
    disp( fullfile( ...
        config.outdir, ...
        prefix, ...
        object_trn_bin_j ));
    
    
    hdf5write( fullfile( ...
        config.outdir, ...
        prefix, ...
        object_trn_bin_j ), ...
        'X', X, ...
        'Y', Y );
    
    fprintf(fid, [ED_meta_line '\n']);
end
fclose(fid);
 
clear binning_info
clear object_data
clear object_score
clear object_X
clear object_Y
clear pos

fid = fopen( fullfile( config.outdir, ...
    prefix, 'info_object_training.txt' ), 'w' );

for k=1:Nobjects
    
    if strcmp( object_classes{ k }, '__background__' )
        continue;
    end
    if strcmp( object_classes{ k }, 'others' )
        continue;
    end
     
    object_data = DataMatrix(use,:);
    
    object_score = object_data(:, OBJ_SCORE_START + k - 1);
    
    pos =  object_score > 0.5 ;
   
    object_data = object_data( pos, :);
   
    object_X = object_data( :, FC7_FEATURES );
    object_Y = object_data( :, OBJ_ATTR );
    
    binning_info = binfun( object_Y ); 
    binning_info(binning_info(:,3) < gamma,:) = [];
        
    if 0 
      
        for j=1:size( binning_info, 1 )
           
            fprintf('%20s | [%d]: %.4f [m] - %.4f [m] | %.4d\n', ...
                object_classes{ k }, ...
                j, ...
                binning_info(j, 1), ...
                binning_info(j, 2), ...
                binning_info(j, 3) );
            
        end
        
    end
    
    for bin_j=1:size( binning_info, 1 )
        
        lo = binning_info( bin_j, 1 );
        hi = binning_info( bin_j, 2 );
        
        ED_meta_line = strcat( object_classes{ k }, ',', ...
            num2str( lo ), ',', ...
            num2str( hi ) );
        
        pos_bin_j =  object_Y >= lo & object_Y < hi ;
        
        object_X_bin_j = object_X( pos_bin_j, : ); % FC7
        object_Y_bin_j = object_Y( pos_bin_j, : ); % Attribute
        
        object_trn_bin_j = sprintf( 'trn_i%.4d.hdf5', bin_j );
        hdf5write( fullfile( ...
                    config.outdir, ...
                    prefix, ...
                    object_classes{ k }, ... 
                    object_trn_bin_j ), ...
                'X', object_X_bin_j, ...
                'Y', object_Y_bin_j );
       
        ED_meta_line = strcat( ED_meta_line, ',', ...
            object_trn_bin_j );
        
        fprintf(fid, [ED_meta_line '\n']);
            
    end
        
end

fclose( fid );






