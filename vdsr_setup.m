function [net,stats] = vdsr_setup(opts,imdb,varargin)
% GitChoi.
% Setups the network structure for VDSR.

net = dagnn.DagNN();

nlr = size(opts.fn,1);

for lr = 1:nlr
    if lr~=1
        net.addLayer(['relu',num2str(lr-1)],dagnn.ReLU(),...
            {['c',num2str(lr-1)]},{['r',num2str(lr-1)]});
    end
    
    if lr==nlr
        net.addLayer(['conv',num2str(lr)],dagnn.Conv('size',opts.fn(lr,:),'pad', (opts.fn(lr,1)-1)/2,'stride', 1),...
            {['r',num2str(lr-1)]},{'res'},{['c',num2str(lr),'f'],['c',num2str(lr),'b']});
        ii = net.getLayerIndex(['conv',num2str(lr)]);
        net.params(net.layers(ii).paramIndexes(1)).learningRate = 1;
        net.params(net.layers(ii).paramIndexes(1)).weightDecay = 1;
        net.params(net.layers(ii).paramIndexes(2)).learningRate = 0.1;
        net.params(net.layers(ii).paramIndexes(2)).weightDecay = 0;
    elseif lr==1
        net.addLayer(['conv',num2str(lr)],dagnn.Conv('size',opts.fn(lr,:),'pad', (opts.fn(lr,1)-1)/2,'stride', 1),...
            {'bic'},{['c',num2str(lr)]},{['c',num2str(lr),'f'],['c',num2str(lr),'b']});
        ii = net.getLayerIndex(['conv',num2str(lr)]);
        net.params(net.layers(ii).paramIndexes(1)).learningRate = 1;
        net.params(net.layers(ii).paramIndexes(1)).weightDecay = 1;
        net.params(net.layers(ii).paramIndexes(2)).learningRate = 0.1;
        net.params(net.layers(ii).paramIndexes(2)).weightDecay = 0;
    else
        net.addLayer(['conv',num2str(lr)],dagnn.Conv('size',opts.fn(lr,:),'pad', (opts.fn(lr,1)-1)/2,'stride', 1),...
            {['r',num2str(lr-1)]},{['c',num2str(lr)]},{['c',num2str(lr),'f'],['c',num2str(lr),'b']});
        ii = net.getLayerIndex(['conv',num2str(lr)]);
        net.params(net.layers(ii).paramIndexes(1)).learningRate = 1;
        net.params(net.layers(ii).paramIndexes(1)).weightDecay = 1;
        net.params(net.layers(ii).paramIndexes(2)).learningRate = 0.1;
        net.params(net.layers(ii).paramIndexes(2)).weightDecay = 0;
    end
end

net.addLayer('Sum',dagnn.Sum(),{'res','bic'},{'prediction'});

net.addLayer('pdist',dagnn.my_Loss_pdist('p',2,'opts',{'noRoot',true}),{'prediction','label'},{'objective'});

net.initParams();

net.print;

[net,stats] = cnn_train_dag_hardclip(net,imdb,getBatch(opts),opts.train);


function fn = getBatch(opts)

bopts = struct('numGpus', numel(opts.train.gpus)) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;


function inputs = getDagNNBatch(opts, imdb, batch)
input = imdb.images.data(:,:,:,batch);
label = imdb.images.label(:,:,:,batch) ;

bic = imresize(input,imdb.sc,'bicubic');

if opts.numGpus > 0
    input = gpuArray(input) ;
    label = gpuArray(label) ;
    bic = gpuArray(bic) ;
end
inputs = {'label', label,'bic',bic} ;




