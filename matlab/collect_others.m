function info = collect_others( data )


Nimages = length( data );

info = containers.Map();

for i=1:Nimages
    
    meta = data(i).groundtruth3DBB;
    
    Ndet = length( meta );
    
    assert( Ndet == length(data(i).groundtruth2DBB) );
    
    for j=1:Ndet
       
        label = meta(j).classname;
        
        if ~isempty(meta(j).labelname)
            continue;
        end
        
        if ~info.isKey(label)
           info(label) = 1; 
        else
           info(label) = info(label)+1;
        end
        
           
    end
    
end

keys = info.keys;
Nocc = zeros(length(keys),1);
for i=1:length(keys)
    Nocc(i) = info(keys{i});
end
[~,idx] = sort(Nocc);
for i=1:length(idx)
   fprintf('%30s [%d]: %.4d\n', keys{idx(i)}, i, info(keys{idx(i)}));
    
end


