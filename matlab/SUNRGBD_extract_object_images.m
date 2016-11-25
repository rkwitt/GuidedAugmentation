function SUNRGBD_extract_object_images( SUNRGBDMeta_correct2D, base, M, object, outdir ) 

if ~exist(outdir, 'dir')
    mkdir(outdir);
end

N = length( SUNRGBDMeta_correct2D );

cnt = 1;
for i=1:N

    meta = SUNRGBDMeta_correct2D(i);
    
    for j=1:length( meta )
       
        if strcmp(meta.groundtruth3DBB(j).classname, object)

            im = fullfile(base, ...
                meta.sequenceName, ...
                'image', ...
                meta.rgbname);
            
            im = imread(im);
            
            name = sprintf('image_%.5d.png', cnt);
            imwrite(im, fullfile(outdir, name));
            cnt = cnt + 1;
            
            if cnt > M;
                return;
            end
            
            %imshow(im);
            %pause;
            %close all;

            
        end
        
    end
    
    
end
