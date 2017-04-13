function [C,M,mC,sC,mM,sM] = SUNRGBD_eval_EDN_performance( data, names, modifier )
% SUNRGBD_eval_EDN_performance.
%   [classRho,classMAE,avgRho,stdRho,medMAE,stdMAE] = 
%   SUNRGBD_eval_EDN_performance(DATA, NAMES, MODIFIER) computes
%   performance measures \rho and MAE for Table 2.

keys = data.keys; % keys specifiy the object class (num. as str.)

C = cell( length( keys ), 1 );
M = cell( length( keys ), 1 );

for i=1:length( keys )
   
    % look at specific object class
    key = keys{i};
    
    % get feature data for the object class
    tmp = data( key );
    
    Ctmp = [];
    Mtmp = [];
    for j=1:size(tmp,1);
        
        org = tmp{j,1}; % original features
        aug = tmp{j,2}; % AGA-augmented features
        inf = tmp{j,3}; % feature information
        
        cc = corr(org', aug'); % compute Pearson rho
        
        % absolute diff. between desired attribute values and attribute
        % values obtained by running the attribute (strength) predictor on
        % AGA-synthesized features.
        df = abs(modifier(inf(:,2))-modifier(inf(:,3))); 
        
        Ctmp = [Ctmp; cc(:)];
        Mtmp = [Mtmp; df(:)];
        
    end
    
    % store results per object class
    C{i} = Ctmp;
    M{i} = Mtmp;
    
end

mC = zeros( length(keys), 1);
sC = zeros( length(keys), 1);
mM = zeros( length(keys), 1);
sM = zeros( length(keys), 1);

for i=1:length( keys )
    mC(i) = mean( C{i} ); % mean over \rho
    sC(i) =  std( C{i} ); % std. dev. over \rho
    
    mM(i) = median( M{i} ); % median absolute error
    sM(i) =  std( M{i} );   % std. dev. over error
    
    % get object name from given cell array of names
    object = names{str2double(keys{i})};
    object = strrep(object, '_', ' ');

    fprintf('%s & %.2f & %.2f \\\\ \n', ...
       object , mC(i), mM(i));
end