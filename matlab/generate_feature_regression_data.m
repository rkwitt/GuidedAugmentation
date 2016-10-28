function generate_feature_regression_data( config, DataMatrix, DataMatrix_img2idx, selection, gamma, Nsplits, seed )

rng( seed );

debug = 1;

OBJ_SCORE_START   = 4100; 
OBJ_DEPTH         = 4098; 
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


fid = fopen( fullfile( config.SUNRGBD_common, 'objects', 'ED_meta.txt' ), 'w' );


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
    
    % OLD BINNING
    %binning_info = binning(object_Y, 5, gamma);
    
    binning_info = binbynum(object_Y, 1, 0.5);
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
            
            object_val_bin_j_cv_k_file = sprintf( 'val_i%.4d_cv_%.4d.hdf5', bin_j, cv_k );
            
            hdf5write( fullfile( ...
                    config.SUNRGBD_common, ...
                    'objects', ...
                    object_classes{ k }, ...
                    object_val_bin_j_cv_k_file ), ...
                'X_val_trn', object_X_bin_j_val_trn, ...
                'X_val_tst', object_X_bin_j_val_tst, ...
                'Y_val_trn', object_Y_bin_j_val_trn, ...
                'Y_val_tst', object_Y_bin_j_val_tst );
            
            ED_meta_line = strcat( ED_meta_line, ',', ...
                object_val_bin_j_cv_k_file );
                
        end
        
        object_trn_bin_j = sprintf( 'trn_i%.4d.hdf', bin_j );
        hdf5write( fullfile( ...
                    config.SUNRGBD_common, ...
                    'objects', ...
                    object_classes{ k }, ... 
                    object_trn_bin_j ), ...
                'X_trn', object_X_bin_j, ...
                'Y_trn', object_Y_bin_j );
       
        ED_meta_line = strcat( ED_meta_line, ',', ...
            object_trn_bin_j );
        
        fprintf(fid, [ED_meta_line '\n']);
            
    end
        
end

fclose( fid );






