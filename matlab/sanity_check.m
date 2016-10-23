labels = {
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

im_file = 'image_00010';

im = imread(['~/Desktop/test/' im_file '.jpg']);
load(['~/Desktop/test/' im_file '_ss_boxes.mat']);
load(['~/Desktop/test/' im_file '_bbox_features.mat']);

for i=1:size(boxes,1)
    
    tmp = CNN_scores(i,:);
    [max_val, max_idx] = max(tmp);
    
    if (max_idx > 2 && max_val > 0.5)
        disp(max_val);
        disp(max_idx);
        disp(labels(max_idx));
        
        box = boxes(i,:);
        x = box(1);
        y = box(2);
        w = floor(box(3)-box(1));
        h = floor(box(4)-box(2));

        imshow(im);
        rectangle('Position', [x y w h], 'LineWidth', 3);
        title(sprintf('%s: %.4f', labels{max_idx}, max_val));
        pause
    end
    
    
end







