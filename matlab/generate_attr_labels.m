
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copy all SUNRGBD images to some common directory
%
% 1. Enumerate all images and rename as image_00001.jpg, ...
% 2. Copy all original image to common directory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


load( fullfile(config.SUNRGBD_dir, 'SUNRGBDMeta_correct2D.mat') );

NImages = size(SUNRGBDMeta_new, 2);

run_image_copy          = 0;
run_selective_search    = 1;
run_labeling            = 0;
run_feature_collection  = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copy all SUNRGBD images to some common directory
%
% 1. Enumerate all images and rename as image_00001.jpg, ...
% 2. Copy all original image to common directory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if run_image_copy

    if ~exist( config.SUNRGBD_common, 'dir' )
        mkdir(out_dir);
    end

    list_file = fullfile( config.SUNRGBD_common, 'allimgs.txt' );

    fid = fopen(list_file, 'w');

    for i=1:NImages

        orgimg_file = fullfile(...
            config.SUNRGBD_dir, ...
            SUNRGBDMeta_new(i).sequenceName, 'image', ...
            SUNRGBDMeta_new(i).rgbname);

        newimg_base = 'image_%.5d';
        newimg_base = sprintf(newimg_base, i);
        newimg_file = fullfile( config.SUNRGBD_common, [newimg_base '.jpg'] );

        fprintf(fid, '%s\n', newimg_base);

        if ~exist( newimg_file, 'file' )

            [success,~,~] = copyfile(...
                orgimg_file, ...
                newimg_file );

        end

    end

    fclose(fid);
    
end
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extracting Selective search bounding boxes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


proplist = cell(NImages, 1);

if run_selective_search

    disp('Selective search bbox extraction');

    for i=1:NImages

        imgfile = fullfile(...
            config.SUNRGBD_common, ...
            sprintf('image_%.5d.jpg', i));
        boxfile = fullfile(...
            config.SUNRGBD_common, ...
            sprintf('image_%.5d_ss_boxes.mat', i));

        % check if proposal file already exists
        if( exist( boxfile, 'file' ) )

            disp([boxfile, ' already exists']);

        else

            % read original image file
            img = imread(imgfile);

            % compute Selective Search proposals
            boxes = selective_search_boxes(img);

            disp(size(boxes));
            disp(boxfile);
            save(boxfile, 'boxes');

            clear boxes img;

        end

        proplist{i} = sprintf('image_%.5d_ss_boxes.mat', i);

    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Load Selective Search bounding boxes
% 2. Annotate them with attributes based on IoU with ground truth boxes
%
% Attributes: Depth from camera plane, x-y plane angle of rotation
%
% For background proposals Attribute labels = -1000
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if run_labeling 

    disp( 'Labeling bounding boxes with attributes' );

    MetaData = {};

    for i=1:NImages

        data = SUNRGBDMeta_new(i);
        disp(['Labeling bounding boxes for image ', num2str(i)]);

        % load proposal file
        load( fullfile( config.SUNRGBD_common, proplist{i} ) );

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

        MetaData(i).attr_depth = attr_depth;
        MetaData(i).attr_angle = attr_angle;
        MetaData(i).prop_label = prop_label;

        clear I_o_u max_* attr_* prop_* boxes data;

    end

end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Collect CNN features, scores and attr labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

det_Classes = {
    '__background__', ...
    'others', ...
    'bathtub', ...
    'bed', ...
    'bookshelf', ...
    'box', ...
    'chair', ...
    'counter', ...
    'desk', ...
    'door', ...
    'dresser', ...
    'garbage bin', ...
    'lamp', ...
    'monitor', ...
    'night stand', ...
    'pillow', ...
    'sink', ...
    'sofa', ...
    'table', ...
    'tv', ...
    'toilet'};


if run_feature_collection
    
    for i=1:length(SUNRGBDMeta_new)
        
        load( fullfile( config.SUNRGBD_common, proplist{i} ) );
        
        MetaData(i).boxes = boxes;
        
        clear boxes;
        
    end
    
    save( fullfile( config.SUNRGBD_common, 'MetaData' ), 'MetaData' );
    
end
    
