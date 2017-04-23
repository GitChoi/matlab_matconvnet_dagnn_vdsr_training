% GitChoi.
% Main training code for VDSR.

close all force
clearvars

opts.train.batchSize = 64;
opts.train.numEpochs = 40;
opts.train.lrstep0 = 20;
opts.train.lrstep = 10;
opts.train.continue = false;
opts.train.gpus = 1;
opts.train.learningRate = 0.1;

mkdir('snap');
opts.train.expDir = fullfile('snap');
opts.imdbPath = fullfile('imdb_vdsr.mat');

load(opts.imdbPath);

nch = 64;
for ly = [20]
    for gc = [1e-4]
        for wd = [1e-4]
            opts.fn = [3 3 1 nch;
                repmat([3 3 nch nch],[ly,1]);
                3 3 nch 1];
            
            opts.train.gclip = gc;
            opts.train.weightDecay = wd;
            
            [net,stats] = vdsr_setup(opts,imdb);
            
            tic
            results = exe_test(opts.train);
            toc
        end
    end
end