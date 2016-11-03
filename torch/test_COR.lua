--Test covariate regressor.

require 'nn'
require 'optim'
require 'hdf5'

-- cmdline parsing
local cmd = torch.CmdLine()
cmd:option('-dataFile',       '/tmp/data.hdf5',       'HDF5 file with features + covariate ground truth')
cmd:option('-outputFile',     '/tmp/output.hdf5',     'HDF5 output file with covariate predictions + ground truth')
cmd:option('-modelCOR',       '/tmp/model_COR.t7',    'Trained COR model')
cmd:option('-column',         1,                      'Column of covariate data [1=Depth, 2=Pose]')
cmd:option('-cuda',           false,                  'Use CUDA')
local opt = cmd:parse(arg)

-- try to use CUDA if required
if opt.cuda then
  require 'cunn'
  require 'cutorch'
end

regressor = torch.load( opt.modelCOR)

if opt.cuda then
  regressor:cuda()
end

local fid = hdf5.open(opt.dataFile, 'r')
local X = fid:read('X'):all()
local Y = fid:read('Y'):all()
fid:close()

--case 1: we only have one input vector
if (X:size():size() == 1) then
  X = X:reshape(1, X:size(1))
  Y = Y:reshape(1, Y:size(1))
else
-- case 2: we have an array of input vectors
  X = X:transpose(1,2)
  Y = Y:transpose(1,2)
end

Y = Y[{{}, opt.column}]

if opt.cuda then
    X = X:cuda()
    Y = Y:cuda()
end

-- MSE loss
local criterion = nn.MSECriterion()
if opt.cuda then
    criterion:cuda()
end

regressor:evaluate()
Y_hat = regressor:forward(X)
Y_err = criterion:forward(Y_hat, Y)  
print('MSE [m]: '..Y_err)
  
fid = hdf5.open(opt.outputFile, 'w')
fid:write('/Y_hat', Y_hat:float())
fid:write('/Y', Y:float())
fid:close()