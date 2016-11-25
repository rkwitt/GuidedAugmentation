function res = one_shot_object_recognition( data, Nruns,  seed )

rng(seed);

Nobjects = length(data.keys);

fprintf('%d objects in one-shot experiment\n', Nobjects);

res = zeros(Nruns,2);

for r=1:Nruns

    Xtrn_ORG = []; % original one-shot samples
    Ytrn_ORG = []; % original one-shot labels
    
    Xtrn_EDN = []; % augmented one-shot samples + original
    Ytrn_EDN = []; % augmented one-shot labels + original

    sel = zeros(Nobjects,1);
    
    % pick one object from each category
    for i=1:Nobjects

        det = data(num2str(i));

        Ndet = size(det,1);

        sel(i,1) = randperm(Ndet,1);

        Xtrn_ORG = [Xtrn_ORG; det{sel(i,1),1}]; %#ok<AGROW>
        Ytrn_ORG = [Ytrn_ORG; i]; %#ok<AGROW>
        
        augmented =  det{sel(i,1),2};
                
        Xtrn_EDN = [Xtrn_EDN; augmented]; %#ok<AGROW>
        Ytrn_EDN = [Ytrn_EDN; ones(size(augmented,1),1)*i];%#ok<AGROW>        
    end
    
    Xtrn_ORG = Xtrn_ORG./repmat(sum(Xtrn_ORG,2),1,4096);
    
    Xtrn_EDN = [Xtrn_EDN; Xtrn_ORG];
    Ytrn_EDN = [Ytrn_EDN; Ytrn_ORG];    
    
    Xtrn_EDN = Xtrn_EDN./repmat(sum(Xtrn_EDN,2),1,4096);
    
    % linear SVM trained on one-shot samples ONLY
    mdl_ORG = train(...
        double(Ytrn_ORG),...
        sparse(double(Xtrn_ORG)), ...
        '-B 1 -c 10 -s 3 -q');
    
    param = '-q -B 1 -s 3 -c 10'; %, num2str(2^bestLog2c)];
    mdl_EDN = train(...
        double(Ytrn_EDN), ...
        sparse(double(Xtrn_EDN)), ...
        param);
    
    Xtst = [];
    Ytst = [];
    for i=1:Nobjects

        det = data(num2str(i));

        Ndet = size(det,1);

        det(sel(i,1),:) = []; % remove one-shot sample

        tmp = zeros(size(det,1),4096);
        for j=1:size(det,1)
            tmp(j,:) = det{j,1};
        end

        Xtst = [Xtst; tmp]; %#ok<AGROW>
        Ytst = [Ytst; ones(Ndet-1,1)*i]; %#ok<AGROW>

    end

    Xtst = Xtst./repmat(sum(Xtst,2),1,4096); % L1 norm

    [lab_ORG, acc_ORG, ~] = predict( ...
        Ytst, ...
        sparse(double(Xtst)), ...
        mdl_ORG, ...
        '-q'); %#ok<ASGLU>
    
    [lab_EDN, acc_EDN, ~] = predict( ...
        Ytst, ...
        sparse(double(Xtst)), ...
        mdl_EDN, ...
        '-q');%#ok<ASGLU>
    

    fprintf('[%d]: Accuracy: %.2f | %.2f\n', r, ...
        acc_ORG(1), ...
        acc_EDN(1));
    
    res(r,1) = acc_ORG(1);
    res(r,2) = acc_EDN(1);
    
    %disp([diag(confusionmat(Ytst,lab_ORG)) ...
    %      diag(confusionmat(Ytst,lab_EDN))]);
    %pause
end


















