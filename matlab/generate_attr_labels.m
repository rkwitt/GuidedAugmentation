
addpath(genpath('.'))

load('/data4/Mandar/SUNRGBD/SUNRGBDtoolbox/Metadata/SUNRGBDMeta_correct2D.mat');

NImages = size(SUNRGBDMeta_new, 2);

dbroot = '/data4/Mandar/';

proplist = cell(NImages, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extracting Selective search bounding boxes
%

disp('Selective search bbox extraction');

for i=1:NImages
    
    data = SUNRGBDMeta_new(i);
    
    rgbfile = data.rgbname;
    seqname = data.sequenceName;
    bboxfile = strrep(rgbfile, '.jpg', '_bbox');
    
    imgfile = fullfile(dbroot, seqname, 'image', rgbfile);
    propfile = fullfile(dbroot, seqname, 'ssbox', bboxfile);
    
    if(exist([propfile, '.mat'], 'file'))
        
        disp([propfile, ' already exists']);
        
    else
    
        img = imread(imgfile);
    
        %xx = size(img, 1); yy = size(img, 2);
        %ss = 512.0/min([xx, yy]);
        %img = imresize(img, ss);
    
        boxes = selective_search_boxes(img);
   
        if(~exist(fullfile(dbroot, seqname, 'ssbox'), 'dir'))
        
            system(['mkdir ', fullfile(dbroot, seqname, 'ssbox')]);
    
        end
    
        disp(['Saving proposal file ', propfile]);
    
        save(propfile, 'boxes');

        clear boxes img;
        
    end
    
    proplist{i} = propfile;
    
    clear data;
    
end
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Selective Search bounding boxes
% Annotate them with attributes based on
% IoU with ground truth boxes
%
% Attributes: Depth from camera plane, x-y plane angle of rotation
%
% For background proposals Attribute labels = -1000
%

disp('Labeling bounding boxes with attributes');

AttrLabels = {};

for i=1:NImages
    
    data = SUNRGBDMeta_new(i);
    
    filename = data.sequenceName;
    
    disp(filename);
    
    load([proplist{i}, '.mat']);
    
    Ngt = size(data.groundtruth2DBB, 2);
    Npr = size(boxes, 1);
    
    I_o_u = zeros(Npr, Ngt);
    
    attr_depth = -1000*ones(Npr, 1);
    attr_angle = -1000*ones(Npr, 1);
    prop_label = cell(Npr, 1);
    
    
    for p =1:Npr
        for k =1:Ngt
            
            bb1 = boxes(p, :);
            bb2 = data.groundtruth2DBB(k).gtBb2D;
            
            if(~isempty(bb2))
                
                bb2(3:4) = bb2(3:4) + bb2(1:2);
            
                I_o_u(p, k) = IoU(bb1, bb2);
            
            end
            
            clear bb*
            
        end
    end
    
    max_ovl = max(I_o_u, [], 2);
    
    max_row = find(max_ovl >= 0.5);
    
    for j = 1:size(max_row, 1)
        
        p = max_row(j, 1);

        [~, gtb] = max(I_o_u(p, :));
        
        attr_depth(p) = data.groundtruth3DBB(gtb).centroid(2);
        attr_angle(p) = acos(data.groundtruth3DBB(gtb).basis(1));
        prop_label{p} = data.groundtruth3DBB(gtb).classname;
    end
    
    AttrLabels(i).attr_depth = attr_depth;
    AttrLabels(i).attr_angle = attr_angle;
    AttrLabels(i).filename = filename;
    AttrLabels(i).prop_label = prop_label;
    
    clear I_o_u max_* attr_* prop_* boxes data;
    
end

save([dbroot, '/SUNRGBD/SUNRGBDtoolbox/Metadata/SUNPoseLabels'], 'AttrLabels');
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Query for Object instances
% Structure with attribute annotations
% and query object labels (1/-1)
% for each proposal
%
query_ob = 'chair';

disp(['Object of choice ', query_ob]);

Obj_AttrLabels = {};

for i=1:NImages
    
    data = SUNRGBDMeta_new(i);
    
    filename = data.sequenceName;
    
    disp(filename);
    
    load(proplist{i});
    
    Npr = size(boxes, 1);
    
    Obj_AttrLabels(i).attr_depth = AttrLabels(i).attr_depth; 
    Obj_AttrLabels(i).attr_angle = AttrLabels(i).attr_angle;
    Obj_AttrLabels(i).filename = filename;
    
    obj_labels = -1*ones(Npr, 1);
    
    for p =1:Npr
    
        if(strcmp(AttrLabels(i).prop_label{p}, query_ob))
           
            obj_labels(p, 1) = 1;
       
        end
        
    end
    
    Obj_AtrrLabels(i).obj_labels = obj_labels; 
    
    clear boxes data obj_labels;

end

save([dbroot, '/SUNRGBD/SUNRGBDtoolbox/Metadata/SUNRGBD_', query_ob, '_PoseLabels'], 'Obj_AttrLabels');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Collect CNN features, scores and attr labels 
% 
%
%
det_Classes = {'background', 'others', 'bathtub', 'bed', 'bookshelf', 'box',...
    'chair', 'counter', 'desk', 'door', 'dresser', 'garbage_bin', 'lamp', 'monitor', ...
    'nght_stand', 'pillow', 'sink', 'sofa', 'table', 'tv', 'toilet'};

%CNN features are currently saved under this dir as
%Features/000001_bbox_features.mat for 1st image in SUNRGBDMeta and so on

vocpath = '~/fast-rcnn/data/SUNRGBDdevkit15/SUNRGBD15/';

for i=1:10335

    img = sprintf('%06d', i);
    
    load([vocpath, '/Features/', img, '_bbox_features']);

    AttrLabels(i).CNN_scores = CNN_scores;
    AttrLabels(i).CNN_feature = CNN_feature;
    
    clear CNN_*

    load([vocpath, '/SelectiveSearch/', img, '_ss_boxes']);
    
    AttrLabels(i).boxes = boxes;
    
    clear boxes;
    
    for p=1:size(AttrLabels(i).proplabel, 2)
        
        if(isempty(AttrLabels(i).proplabel{p})
              AttrLabels(i).proplabel21(p) = 1;
        else
            
            IndexC = strfind(det_Classes, ATtrLabels(i).proplabel{p});
            Index = find(not(cellfun('isempty', IndexC)));
            
            if(isempty(Index))
                AttrLabels(i).proplabel21(p) = 2;
            else
                AttrLabels(i).proplabel21(p) = Index;
            end
                clear Index*
                
        end
        
    end
    
    disp(img);
    
end

save([dbroot, '/SUNRGBD/SUNRGBDtoolbox/Metadata/SUNRGBDMeta_fastRCNN_21cls'], 'AttrLabels');
