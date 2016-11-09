function [Xtrn, Ytrn] = one_shot_object_recognition( data, seed )

rng(seed);

Nobjects = length(data.keys);

%
% one-shot
%
addpath('~/Documents/MATLAB/liblinear-1.94/matlab/');

Xtrn = [];
Ytrn = [];

sel = zeros(Nobjects,1);
for i=1:Nobjects
   
    det = data(num2str(i));
    
    Ndet = size(det,1);
    
    sel(i,1) = randperm(Ndet,1);
    
    Xtrn = [Xtrn; det{sel(i,1),1}];
    Ytrn = [Ytrn; i];
    
end

mdl = train(double(Ytrn),sparse(double(Xtrn)),'-s 3 -c 2 -B 1');


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
    
    Xtst = [Xtst; tmp];
    Ytst = [Ytst; ones(Ndet-1,1)*i];
    
end

[~, ac, ~] = predict(Ytst, sparse(double(Xtst)), mdl);


