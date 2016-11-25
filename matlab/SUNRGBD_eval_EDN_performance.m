function [C,M,mC,sC,mM,sM] = SUNRGBD_eval_EDN_performance( data, names, modifier ) 

keys = data.keys;

C = cell( length( keys ), 1 );
M = cell( length( keys ), 1 );

for i=1:length( keys )
   
    key = keys{i};
    
    tmp = data( key );
    
    Ctmp = [];
    Mtmp = [];
    for j=1:size(tmp,1);
        
        org = tmp{j,1};
        aug = tmp{j,2};
        inf = tmp{j,3};
        
        cc = corr(org', aug');
        df = abs(modifier(inf(:,2))-modifier(inf(:,3)));
        
        Ctmp = [Ctmp; cc(:)];
        Mtmp = [Mtmp; df(:)];
        
    end
    
    C{i} = Ctmp;
    M{i} = Mtmp;
    
end

mC = zeros( length(keys), 1);
sC = zeros( length(keys), 1);
mM = zeros( length(keys), 1);
sM = zeros( length(keys), 1);

for i=1:length( keys )
   
    mC(i) = mean( C{i} );
    sC(i) =  std( C{i} );
    
    mM(i) = mean( M{i} );
    sM(i) =  std( M{i} );
    
    object = names{str2double(keys{i})};
    object = strrep(object, '_', ' ');

    fprintf('%s & %.2f & %.2f \\\\ \n', ...
       object , mC(i), mM(i));
    
end









