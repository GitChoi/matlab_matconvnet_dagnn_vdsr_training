classdef my_Loss_pdist < dagnn.ElementWise
    % Edited by GitChoi.
    % A pdist variant for DagNN. Used as a loss layer for regression.
    
    properties
        p = 2;
        opts = {}
    end
    
    properties (Transient)
        average = 0
        numAveraged = 0
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = my_vl_nnpdist(inputs{1}, inputs{2},obj.p,obj.opts{:});
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + double(gather(outputs{1}))/(size(inputs{1},1)*size(inputs{1},2)*size(inputs{1},3))*(21*21)) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = my_vl_nnpdist(inputs{1}, inputs{2},obj.p, derOutputs{1}, obj.opts{:}) ;
            derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function reset(obj)
            obj.average = 0 ;
            obj.numAveraged = 0 ;
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
            outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
        end
        
        function rfs = getReceptiveFields(obj)
            % the receptive field depends on the dimension of the variables
            % which is not known until the network is run
            rfs(1,1).size = [NaN NaN] ;
            rfs(1,1).stride = [NaN NaN] ;
            rfs(1,1).offset = [NaN NaN] ;
            rfs(2,1) = rfs(1,1) ;
        end
        
        function obj = my_Loss_pdist(varargin)
            obj.load(varargin) ;
        end
    end
end
