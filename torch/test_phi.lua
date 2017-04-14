-- test encoder-decoder network (phi) on new data

require 'optim'
require 'hdf5'
require 'nn'


-- cmdline parsing
local cmd = torch.CmdLine()
cmd:option('-dataFile', 			'/tmp/data.hdf5', 			'HDF5 data file')
cmd:option('-outputFile', 			'/tmp/output.hdf5', 		'HDF5 output file')
cmd:option('-model',                '/tmp/model.t7',			'Trained phi+gamma model')
cmd:option('-cuda', 				false, 						'Use CUDA')

local opt = cmd:parse(arg)

-- try to use CUDA if possible
if opt.cuda then
	require 'cunn'
	require 'cutorch'
end

model = torch.load(opt.model)
modelPhi   = model:get(1)
modelGamma = model:get(2):get(2)

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

-- run data through phi+gamma to get a prediction
-- for synthesized features.
model:evaluate()
local Y_hat_PhiGamma = model:forward(X)[2]
print(torch.mean(Y_hat_PhiGamma))

-- run data through phi to get synthesized features.
modelPhi:evaluate()
local X_hat = modelPhi:forward(X)

fid = hdf5.open(opt.outputFile, 'w')
fid:write('/X', X:float() )                      			-- x
fid:write('/X_hat', X_hat:float())							-- phi(x)
fid:write('/Y_hat_PhiGamma', Y_hat_PhiGamma:float())		-- gamma(phi(x))
fid:write('/Y_hat_Gamma', modelGamma:forward(X):float())	-- gamma(x)
fid:close()