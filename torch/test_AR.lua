--Test attribute regressor.

require 'nn'
require 'optim'
require 'hdf5'

-- cmdline parsing
local cmd = torch.CmdLine()
cmd:option('-dataFile',       '/tmp/data.hdf5',       'HDF5 file with features + covariate ground truth')
cmd:option('-outputFile',     '/tmp/output.hdf5',     'HDF5 output file with covariate predictions + ground truth')
cmd:option('-model',          '/tmp/model_AR.t7',     'Trained AR model')
cmd:option('-column',         1,                      'Column of covariate data [1=Depth, 2=Pose]')
cmd:option('-eval',           false,                  'Evaluate (requires ground truth in data file)')
cmd:option('-cuda',           false,                  'Use CUDA')
local opt = cmd:parse(arg)

-- try to use CUDA if required
if opt.cuda then
  require 'cunn'
  require 'cutorch'
end

regressor = torch.load( opt.model)

if opt.cuda then
  regressor:cuda()
end

local fid = hdf5.open(opt.dataFile, 'r')
local X = fid:read('X'):all()
local Y = nil
if opt.eval then
  Y = fid:read('Y'):all()
end
fid:close()

--case 1: we only have one input vector
if (X:size():size() == 1) then
  X = X:reshape(1, X:size(1))
  if opt.eval then
    Y = Y:reshape(1, Y:size(1))
  end
else
-- case 2: we have an array of input vectors
  X = X:transpose(1,2)
  if opt.eval then
    Y = Y:transpose(1,2)
  end
end

if opt.eval then
  Y = Y[{{}, opt.column}]
end

if opt.cuda then
    X = X:cuda()
    if opt.eval then
      Y = Y:cuda()
    end
end

-- MSE loss
local criterion = nn.MSECriterion()
if opt.cuda then
    criterion:cuda()
end

-- switch to evaluation mode
regressor:evaluate()
Y_hat = regressor:forward(X)
Y_err = nil
if opt.eval then
  Y_err = criterion:forward(Y_hat, Y)  
  print('MSE [m]: '..Y_err)
end

-- write predictions and ground-truth for easy evaluation
fid = hdf5.open(opt.outputFile, 'w')
fid:write('/Y_hat', Y_hat:float())
if opt.eval then
  fid:write('/Y', Y:float())
end
fid:close()