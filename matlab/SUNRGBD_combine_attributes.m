function data = SUNRGBD_combine_attributes( depth, pose )

Nobjects_depth = length( depth.keys );
Nobjects_pose  = length( pose.keys );

assert( Nobjects_pose == Nobjects_depth );

data = containers.Map();

for i=1:Nobjects_pose

        key = num2str(i);
    
        det_depth = depth( key );
        det_pose = pose( key );
        
        Ndet_depth = size(det_depth,1);
        Ndet_pose = size(det_pose,1);
        
        assert( Ndet_depth == Ndet_pose );
        
        tmp = cell(Ndet_depth, 2);
        for j=1:Ndet_depth
        
            X_depth = det_depth{j,2}; % augmented depth-features
            X_pose  = det_pose{j,2};  % augmented pose-features
            X = [X_depth; X_pose];    % augmented pose+depth features
            
            tmp{j,1} = det_depth{j,1};
            tmp{j,2} = X;
            tmp{j,3} = []; % dummy
            
        end
        
        data( key ) = tmp;
        
end





