function E = binbystep(x, delta, overlap)

max_x = max(x);

start = 0;

E = [];
while( start + delta <= max_x );

    lo = start;
    hi = start + delta;
    
    N = length(find( x >= lo & x < hi ) );
    
    tmp = [lo hi N];
    
    E = [E; tmp]; %#ok<AGROW>
    
    start = start + overlap;
    
end

% account for last interval (which might be smaller than delta)
lo = start;
hi = start + delta;
tmp = [lo hi length(find( x >= lo & x < hi ) )];
E = [E; tmp];
