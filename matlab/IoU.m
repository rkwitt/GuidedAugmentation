function[I] = IoU(box1, box2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function[I] = IoU(box1, box2)
%
% Input: Coordinates of bounding box extremities (diagonally opposite
% corners)
% Output: Intersection over Union


xx1 = max(box1(1), box2(1));
xx2 = min(box1(3), box2(3));

yy1 = max(box1(2), box2(2));
yy2 = min(box1(4), box2(4));

if((xx2 < xx1) || (yy2 < yy1))
    
    %disp('No overlap');
    I = 0;
    
else
    
    A1 = (box1(3) - box1(1))*(box1(4) - box1(2));
    A2 = (box2(3) - box2(1))*(box2(4) - box2(2));
    I = (xx2 - xx1)*(yy2 - yy1);
    
    I = I/(A1 + A2 - I);
    
end
