function E = binning(x, N, gamma)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create a binning of the values in x into N bins and keep bins that have
% at least gamma observations.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[num, edges, bin] = histcounts( x, N );

pos = find( num > gamma );

Npos = length( pos );

E = zeros( Npos, 3 );

for j=1:length(pos)

    idx = pos( j );
    
    E(j, 1) = edges( idx );                         % L-boundary
    E(j, 2) = edges( idx+1 );                       % R-boundary
    E(j, 3) = length( find( bin == pos( j ) ) );    % #samples in range
    
end

