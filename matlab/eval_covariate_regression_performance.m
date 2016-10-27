function eval_covariate_regression_performance( config )

load( 'object_classes' );

Nobjects = length( object_classes );

for i=3:Nobjects % skip __background__ + others
   
    object = object_classes{i};
    
    eval_file = fullfile( ...
        config.SUNRGBD_common, ...
        'objects', ...
        object, ...
        'evaluation.hdf5' );
    
    
    y_hat = hdf5read( eval_file, 'y_hat' );
    y_tst = hdf5read( eval_file, 'y_tst' );
    
    y_hat = y_hat(:);
    y_tst = y_tst(:);
    
    mse = mean( ( y_hat - y_tst ).^2 ) ;
    
    fprintf('%20s | MSE=%.4f [m]\n', object, mse );
     
end