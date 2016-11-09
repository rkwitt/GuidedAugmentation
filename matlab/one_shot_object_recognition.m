function one_shot_object_recognition( data, Nruns,  seed )

rng(seed);

Nobjects = length(data.keys);

fprintf('%d objects in one-shot experiment\n', Nobjects);

addpath('~/Documents/MATLAB/liblinear-1.94/matlab/');

for r=1:Nruns

    Xtrn_ORG = []; % original one-shot samples
    Ytrn_ORG = []; % original one-shot labels
    
    Xtrn_EDN = []; % augmented one-shot samples + original
    Ytrn_EDN = []; % augmented one-shot labels + original

    sel = zeros(Nobjects,1);
    for i=1:Nobjects

        det = data(num2str(i));

        Ndet = size(det,1);

        sel(i,1) = randperm(Ndet,1);

        Xtrn_ORG = [Xtrn_ORG; det{sel(i,1),1}]; %#ok<AGROW>
        Ytrn_ORG = [Ytrn_ORG; i]; %#ok<AGROW>
        
        augmented =  det{sel(i,1),2};
        augmented = augmented(1:2,:);
        
        Xtrn_EDN = [Xtrn_EDN; augmented]; %#ok<AGROW>
        Ytrn_EDN = [Ytrn_EDN; ones(size(augmented,1),1)*i];%#ok<AGROW>
        Xtrn_EDN = [Xtrn_EDN]; %#ok<AGROW>
        Ytrn_EDN = [Ytrn_EDN]; %#ok<AGROW>
        
    end
    
    % linear SVM trained on one-shot samples ONLY
    mdl_ORG = train(...
        double(Ytrn_ORG),...
        sparse(double(Xtrn_ORG)), ...
        '-B 1 -q');
    
    % linear SVM trained on one-shot samples + augmented data
    mdl_EDN = train(...
        double(Ytrn_EDN), ...
        sparse(double(Xtrn_EDN)), ...
        '-B 1 -q');
    
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

    [lab_ORG, acc_ORG, ~] = predict( ...
        Ytst, ...
        sparse(double(Xtst)), ...
        mdl_ORG, ...
        '-q');
    
    [lab_EDN, acc_EDN, ~] = predict( ...
        Ytst, ...
        sparse(double(Xtst)), ...
        mdl_EDN, ...
        '-q');
    
    fprintf('Accuracy: %.2f | %.2f\n', ...
        acc_ORG(1), ...
        acc_EDN(1));
    
    disp([diag(confusionmat(Ytst,lab_ORG)) ...
          diag(confusionmat(Ytst,lab_EDN))]);
    pause
end


















