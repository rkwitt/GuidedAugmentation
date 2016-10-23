function[I] = IoU(box1, box2)
% IoU computes the Intersection-Over-Union metric between two bounding
% boxes.
%   I = IoU(BOX1, BOX2) computes the Intersection-Over-Union metric between
%   BOX1 and BOX2. The boxes are specified as:
%
%   x1,y1,x2,y2 - diagonally opposite corners
%
%   CAUTION: this specification is different from MATLABs bboxOverlapRatio
%   which expects boxes in the format [x y w h].s
%
% Mandar Dixit, Roland Kwitt, 2016

xx1 = max(box1(1), box2(1));
xx2 = min(box1(3), box2(3));

yy1 = max(box1(2), box2(2));
yy2 = min(box1(4), box2(4));

if((xx2 < xx1) || (yy2 < yy1))

    I = 0;

else
    
    A1 = (box1(3) - box1(1))*(box1(4) - box1(2));
    A2 = (box2(3) - box2(1))*(box2(4) - box2(2));
    I = (xx2 - xx1)*(yy2 - yy1);
    
    I = I/(A1 + A2 - I);
    
end
