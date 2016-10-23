-- autoencoder+regularizer, author: rkwitt, mdixit (2016)
require 'optim'
require 'hdf5'
require 'nn'

-- cmdline parsing
local cmd = torch.CmdLine()
-- input/output files
cmd:option('-dataFile', '/tmp/data.hdf5', '(Input) HDF5 source data file')
cmd:option('-validationFile', '/tmp/validation.hdf5', '(Input) HDF5 validation file')
cmd:option('-predictionFile', '/tmp/prediction.hdf5', '(Output) HDF5 prediction file')
cmd:option('-logFile', '/tmp/adjuster.log', '(Output) logfile')
-- models
cmd:option('-autoencoderModelFile', '/tmp/autoencoder.hdf5', '(Input) pre-trained autoencoder file')
cmd:option('-regressorModelFile', '/tmp/regressor.t7', 'Trained covariate regressor')
-- misc. options
cmd:option('-predict', false, 'Predict activations of validation file')
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
local src = fid:read('X_source'):all():transpose(1,2)
fid:close()

local N = src:size(1) -- nr. of data points
local D = src:size(2) -- nr. of dimensions
print('#Source data points: '..N..'x'..D)

-- load pre-trained regressor and freeze layers
local regressor = torch.load(opt.regressorModelFile)
for i, m in ipairs(regressor.modules) do
  m.accGradParameters = function() end
  m.updateParameters  = function() end
end
print('Froze covariate regressor layers')

-- load pre-trained autoencoder
ae = torch.load(opt.autoencoderModelFile)

-- build  the final model
model = nn.Sequential()
model:add(ae)
model:add(regressor)
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
  -- backpropagate
  model:backward(x, grad_loss)
  logger:add({'opfunc [loss]: '..loss})
  return loss, gradTheta
end


for epoch = 1, opt.epochs do
  -- random indices for minibatches
  local indices = torch.randperm(N):long():split(opt.batchSize)
  indices[#indices] = nil
  -- run over minibatches
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)
    x = src:index(1, v) -- batch src
    y = torch.Tensor(x:size(1),1):fill(opt.target)
    if opt.cuda then
      x = x:cuda()
      y = y:cuda()
    end
    tmp, batch_loss = optim.adam(opfunc, theta, config)
  end
end

if opt.predict then
  -- load validation data
  fid = hdf5.open(opt.validationFile, 'r')
  local val = fid:read('X_validation'):all():transpose(1,2)
  fid:close()

  if opt.cuda then
    val = val:cuda()
  end

  model:evaluate()
  local prediced_covariate = model:forward(val)
  -- DEBUG purposes
  print(prediced_covariate)
  print(torch.mean(prediced_covariate))

  ae:evaluate()
  local predicted_activations = ae:forward(val)
  fid = hdf5.open(opt.predictionFile, 'w')
  fid:write('/predicted_activations', predicted_activations:float())
  fid:write('/predicted_covariate', prediced_covariate:float())
  fid:write('/initial_covariate', regressor:forward(val):float())
  fid:close()
end
