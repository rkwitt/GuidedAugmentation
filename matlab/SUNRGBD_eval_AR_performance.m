function [mae_agnostic, mae_object] = SUNRGBD_eval_AR_performance( base, prefix, modifier )
% EVAL_AR_PERFORMANCE. 
%   [MAE_AGNOSTIC, MAE_OBJECT] = SUNRGBD_eval_AR_performance(BASE, PREFIX,
%   MODIFIER) reports the performance of attribute regressors (ARs), trained 
%   in an object-specific (MAE_OBJECT) and object-agnostic manner
%   (MAE_AGNOSTIC).
%
% Input:
%
%   BASE        ... Base folder
%   PREFIX      ... Model folder
%   MODIFIER    ... Error modifier, e.g., @(x) rad2deg(x) for pose
%
%   Example directory structure:
%
%       /BASE/PREFIX/chair
%       /BASE/PREFIX/bed 

load( 'matfiles/SUNRGBD_objects' );

Nobjects = length( object_classes ); %#ok<USENS>

mse_agnostic = zeros(Nobjects-2,1);
mse_object = zeros(Nobjects-2,1);

mae_agnostic = zeros(Nobjects-2,1);
mae_object = zeros(Nobjects-2,1);

cnt = 1;
for i=3:Nobjects % skip __background__ + others
   
    object = object_classes{i};
    
    % read object agnostic attribute predictions
    eval_file = fullfile( ...
        base, ...
        prefix, ...
        object, ...
        'agnosticAR_predictions.hdf5' );
    y_hat_agnostic = hdf5read( eval_file, 'Y_hat' );
    y_tst_agnostic = hdf5read( eval_file, 'Y' );
    y_hat_agnostic = y_hat_agnostic(:);
    y_tst_agnostic = y_tst_agnostic(:);
    
    % compute mean and median absolute error
    mse_agnostic(cnt) = mean( ( y_hat_agnostic - y_tst_agnostic ).^2 ) ;
    mae_agnostic(cnt) = median( abs( y_hat_agnostic - y_tst_agnostic ) ) ;
    
    % read object-specific attribute predictions
    eval_file = fullfile( ...
        base, ...
        prefix, ...
        object, ...
        'objectAR_predictions.hdf5' );
    y_hat_object = hdf5read( eval_file, 'Y_hat' );
    y_tst_object = hdf5read( eval_file, 'Y' );
    y_hat_object = y_hat_object(:);
    y_tst_object = y_tst_object(:);
    
    % compute mean and median absolute error
    mse_object(cnt) = mean( ( y_hat_object - y_tst_object ).^2 ) ;
    mae_object(cnt) = median( abs( y_hat_object - y_tst_object ) );
    
    fprintf('%20s & %.2f & %.2f \\\\ \n', ...
        object, modifier(mae_object(cnt)), modifier(mae_agnostic(cnt) ));

    cnt = cnt + 1;
end