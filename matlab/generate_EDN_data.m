function generate_EDN_data( config, DataMatrix, DataMatrix_img2idx, selection, gamma, Nsplits, seed )
%
% Generate data for EDN training
%

rng( seed );

debug = 1;

OBJ_SCORE_START   = 4100; 
OBJ_DEPTH         = 4098; 
OBJ_ANGLE         = 4099; 
FC7_FEATURES      = 1:4096; 

load( 'object_classes' );

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


fid = fopen( fullfile( config.SUNRGBD_common, 'objects', 'info_agnostic_training.txt' ), 'w' );

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
object_Y = object_data( :, OBJ_DEPTH );
object_L = object_data( :, OBJ_SCORE_START+21-1 );
    
binning_info = binbystep(object_Y, 1, 0.5);
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
         pp = randsample(find(object_L_bin_j(:,m)==m), 200);
         X = [X; object_X_bin_j(pp,:)];
         Y = [Y; object_Y_bin_j(pp,:)];
     end
     
     object_trn_bin_j = sprintf( 'trn_i%.4d.hdf5', bin_j );
     
     ED_meta_line = strcat(num2str( lo ), ',', ...
         num2str( hi ), ',', object_trn_bin_j );
     
     hdf5write( fullfile( ...
         config.SUNRGBD_common, ...
         'objects', ...
         object_trn_bin_j ), ...
         'X', X, ...
         'Y', Y );
     
         %'X', object_X_bin_j, ...
         %'Y', object_Y_bin_j );
     
     fprintf(fid, [ED_meta_line '\n']); 
 end
 fclose(fid);

 
clear binning_info
clear object_data
clear object_score
clear object_X
clear object_Y
clear pos


fid = fopen( fullfile( config.SUNRGBD_common, 'objects', 'info_object_training.txt' ), 'w' );

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
    object_Y = object_data( :, OBJ_DEPTH );
    
    binning_info = binbystep(object_Y, 1, 0.5);
    binning_info(binning_info(:,3) < gamma,:) = [];
        
    if debug 
      
        for j=1:size( binning_info, 1 )
           
            fprintf('%20s | [%d]: %.4f [m] - %.4f [m] | %.4d\n', ...
                object_classes{ k }, ...
                j, ...
                binning_info(j, 1), ...
                binning_info(j, 2), ...
                binning_info(j, 3) );
            
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % For each bin, we 
    %
    %   1. split the data for each bin into validation train/test T times
    %   2. store the data for each bin as it is
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for bin_j=1:size( binning_info, 1 )
        
        lo = binning_info( bin_j, 1 );
        hi = binning_info( bin_j, 2 );
        
        ED_meta_line = strcat( object_classes{ k }, ',', ...
            num2str( lo ), ',', ...
            num2str( hi ) );
        
        pos_bin_j =  object_Y >= lo & object_Y < hi ;
        
        object_X_bin_j = object_X( pos_bin_j, : ); % FC7
        object_Y_bin_j = object_Y( pos_bin_j, : ); % Attribute
        
        % evaluation cv splits
        for cv_k=1:Nsplits
        
            [val_trn, val_tst] = crossvalind('HoldOut', size( object_X_bin_j, 1 ), 0.2 );

            object_X_bin_j_val_trn = object_X_bin_j( val_trn, : );
            object_X_bin_j_val_tst = object_X_bin_j( val_tst, : );
            object_Y_bin_j_val_trn = object_Y_bin_j( val_trn, : );
            object_Y_bin_j_val_tst = object_Y_bin_j( val_tst, : );
            
            object_val_bin_j_cv_k_file = sprintf( 'val_i%.4d_cv_%.4d', bin_j, cv_k );
            
            hdf5write( fullfile( ...
                    config.SUNRGBD_common, ...
                    'objects', ...
                    object_classes{ k }, ...
                    strcat( object_val_bin_j_cv_k_file, '_train.hdf5') ), ...
                'X', object_X_bin_j_val_trn, ...
                'Y', object_Y_bin_j_val_trn );
                
            hdf5write( fullfile( ...
                    config.SUNRGBD_common, ...
                    'objects', ...
                    object_classes{ k }, ...
                    strcat( object_val_bin_j_cv_k_file, '_test.hdf5') ), ...
                'X', object_X_bin_j_val_tst, ...
                'Y', object_Y_bin_j_val_tst );
            
            ED_meta_line = strcat( ED_meta_line, ',', ...
                object_val_bin_j_cv_k_file );
                
        end
        
        object_trn_bin_j = sprintf( 'trn_i%.4d.hdf5', bin_j );
        hdf5write( fullfile( ...
                    config.SUNRGBD_common, ...
                    'objects', ...
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





