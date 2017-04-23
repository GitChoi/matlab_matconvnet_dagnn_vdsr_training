% GitChoi
% Creates an imdb dataset for training

close all force
clearvars

sc = 2;
sz = 16;
stride = 16;

yl = single([]);
yh = single([]);
yb = single([]);

dirname = fullfile('your_training_folder_path');

img_dir = dir(fullfile(dirname,'*.bmp'));

Nas = 1;
Nar = 4;
Naf = 2;

h = waitbar(0,'Please wait...');
steps = length(img_dir)*Nas*Nar*Naf;
step = 0;

nnt = 0;
for i1 = 1:length(img_dir)
    aug2{1} = imread(fullfile(dirname,img_dir(i1).name));
    
    for i2 = 1:Nas
        aug3 = aug2{i2};
        aug3 = rgb2ycbcr(aug3);
        aug3 = aug3(:,:,1);
        aug3 = single(aug3);
        sh = size(aug3);
        sh = floor(sh/sc)*sc;
        aug3 = aug3(1:sh(1),1:sh(2));
        
        for i3 = 1:Nar
            aug4 = imrotate(aug3,90*(i3-1));
            
            for i4 = 1:Naf
                if i4 == 2
                    aug4 = fliplr(aug4);
                end
                
                step = step+1;
                waitbar(step/steps);
                
                imh = aug4;
                sh = size(imh);
                
                iml = imresize(imh,1/sc,'bicubic');
                
                imlv = extract_subim(iml,sz,stride);
                imhv = extract_subim(imh,sz*sc,stride*sc);
                
                len = size(imlv,2);
                
                yl(:,end+1:end+len) = imlv;
                yh(:,end+1:end+len) = imhv;
            end
        end
    end
end
close(h)

nim = size(yl,2);
indpick = randperm(nim);

yl = yl(:,indpick);
yh = yh(:,indpick);

yh = yh/255;
yl = yl/255;

yl = reshape(yl,[sz,sz,1,nim]);
yh = reshape(yh,[sz*sc,sz*sc,1,nim]);

clear imdb

imdb.images.data = yl;
imdb.images.label = yh;
imdb.images.set = ones(1,nim);
imdb.sc = sc;
imdb.sz = sz;
imdb.stride = stride;
imdb.nim = nim;

save('imdb_vdsr','imdb','-v7.3');