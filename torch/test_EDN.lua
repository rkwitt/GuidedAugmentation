-- test encoder-decoder network (EDN) on new data
require 'optim'
require 'hdf5'
require 'nn'

-- cmdline parsing
local cmd = torch.CmdLine()
cmd:option('-dataFile', 			'/tmp/data.hdf5', 			'HDF5 data file')
cmd:option('-outputFile', 			'/tmp/output.hdf5', 		'HDF5 output file')
cmd:option('-model',                '/tmp/model.t7',			'Trained encoder-decoder network with regressor (EDN-COR)')
cmd:option('-cuda', 				false, 						'Use CUDA')

local opt = cmd:parse(arg)

-- try to use CUDA if possible
if opt.cuda then
	require 'cunn'
	require 'cutorch'
end

model = torch.load(opt.model)
modelEDN = model:get(1) -- encoder-decoder network
modelCOR = model:get(2) -- covariate regressor
--print(model)

if opt.cuda then
	model = model:cuda()
end

-- load data
local fid = hdf5.open(opt.dataFile, 'r')
local X = fid:read('X'):all()

--case 1: we only have one input vector
if (X:size():size() == 1) then
  X = X:reshape(1, X:size(1))
else
-- case 2: we have an array of input vectors
  X = X:transpose(1,2)
end

if opt.cuda then
	X = X:cuda()
end

model:evaluate()
local Y_hat_EDNCOR = model:forward(X)
--print(torch.mean(Y_hat_EDNCOR))
 
modelEDN:evaluate()
local X_hat = modelEDN:forward(X)

fid = hdf5.open(opt.outputFile, 'w')
fid:write('/X', X:float() )                      	
fid:write('/X_hat', X_hat:float())
fid:write('/Y_hat_EDNCOR', Y_hat_EDNCOR:float())
fid:write('/Y_hat_COR', modelCOR:forward(X):float())
fid:close()