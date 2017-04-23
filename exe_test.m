function results = exe_test(opt)
% GitChoi
% Testing code for VDSR

dirpath = 'your_testing_images_folder_path';
img_dir = dir(fullfile(dirpath,'*.bmp'));

sc = 2;

results = [];
for iter = 2:opt.numEpochs
    load([opt.expDir,'/net-epoch-',num2str(iter),'.mat'],'net','stats');
    
    net2 = dagnn.DagNN.loadobj(net);
    net2.mode = 'test';
    net2.move('gpu');
    net2.vars(net2.getVarIndex('prediction')).precious = true;
    
    clear out_im out_psnr
    for i = 1:length(img_dir)
        imh = imread(fullfile(dirpath,img_dir(i).name));
        imh = rgb2ycbcr(imh);
        imh = imh(:,:,1);
        imh = single(imh);
        sh = size(imh);
        sh = floor(sh/sc)*sc;
        imh = imh(1:sh(1),1:sh(2));
        iml = imresize(imh,1/sc,'bicubic');
        imb = imresize(iml,sc,'bicubic');
        
        tic
        
        net2.eval({'input',gpuArray(iml/255),'bic',gpuArray(imb/255)});
        
        ims = net2.vars(net2.getVarIndex('prediction')).value;
        ims = gather(ims);
        
        tt = toc;
        
        ims = ims*255;
        
        if iter == opt.numEpochs
            figure; imshow(uint8(imh));
            figure; imshow(uint8(imb));
            figure; imshow(uint8(ims));
        end
        out_psnr(i) = psnr(double(ims),double(imh),255);
        Ttt(i) = tt;
    end
    
    results(1,iter) = mean(out_psnr);
    results(2,iter) = mean(Ttt);
end
results(1,1) = results(1,2);
figure; plot(results(1,:));