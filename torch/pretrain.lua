-- pretraining of AE part --
require 'torch'
require 'optim'
require 'hdf5'
require 'nn'

-- cmdline parsing
local cmd = torch.CmdLine()
cmd:option('-logFile', '/tmp/pretrain.log', 'Logfile')
cmd:option('-dataFile', '/tmp/data.hdf5', '(Input) HDF5 source data file')
cmd:option('-modelFile', 'torch/autoencoder.lua')
cmd:option('-saveModel', '/tmp/autoencoder.hdf5', '(Output) Save model to file')
cmd:option('-learningRate', 0.001, 'Learning rate')
cmd:option('-target', 4, 'Target covariate')
cmd:option('-epochs', 10, 'Training epochs')
cmd:option('-batchSize', 300, 'Batchsize')
cmd:option('-cuda', false, 'Use CUDA')
local opt = cmd:parse(arg)

-- try to use CUDA if possible
if opt.cuda then
	require 'cunn'
	require 'cutorch'
end

-- logger
logger = optim.Logger(opt.logFile)

-- load some training/testing data
local fid = hdf5.open(opt.dataFile, 'r')
local src = fid:read('X_trn'):all():transpose(1,2)
fid:close()

local N = src:size(1) -- nr. of data points
local D = src:size(2) -- nr. of dimensions
print('#Source data points: '..N..'x'..D)

-- load model
model = dofile(opt.modelFile)
print(model)
if opt.cuda then
  model:cuda()
end

-- configure optimizer
local config = { learningRate = opt.learningRate }
print(config)

-- model paramters
local theta, gradTheta = model:getParameters()

-- MSE loss
local criterion = nn.MSECriterion()
if opt.cuda then
  criterion:cuda()
end

local x -- minibatch src
local y -- minibatch dst

-- optimization function
local opfunc = function(params)
  if theta ~= params then
    theta:copy(params)
  end
  -- zero model gradients
  gradTheta:zero()
  -- forward pass through model
  local x_hat = model:forward(x)
  -- compute loss
  local loss = criterion:forward(x_hat, y)
  -- gradient wrt loss
  local grad_loss = criterion:backward(x_hat, y)
  -- backprop
  model:backward(x, grad_loss)
  logger:add({'opfunc [loss]: '..loss})
  return loss, gradTheta
end

-- run over opt.epochs epochs
for epoch = 1, opt.epochs do
  -- random indices for minibatches
  local indices = torch.randperm(N):long():split(opt.batchSize)
  indices[#indices] = nil
  -- run over minibatches
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)
    x = src:index(1, v) -- batch src
    y = src:index(1, v) -- batch dst
    if opt.cuda then
    	x = x:cuda()
    	y = y:cuda()
    end
    tmp, batch_loss = optim.adam(opfunc, theta, config)
  end
end

torch.save(opt.saveModel, model)
