require 'nn'
require 'optim'
require 'hdf5'

-- cmdline parsing
local cmd = torch.CmdLine()
cmd:option('-logFile', '/tmp/regressor.log', 'Logfile')
cmd:option('-evaluationFile', '/tmp/evaluation.hdf5', 'File with predictions + ground truth')
cmd:option('-trainFile', '/tmp/train.hdf5', '(Input) HDF5 training data file')
cmd:option('-testFile', '/tmp/test.hdf5', '(Input) HDF5 testing data file')
cmd:option('-test', false, 'Run on test file')
cmd:option('-target', 1, '1=Depth, 2=Pose')
cmd:option('-learningRate', 0.001, 'Learning rate')
cmd:option('-epochs', 10, 'Training epochs')
cmd:option('-batchSize', 300, 'Batchsize')
cmd:option('-saveModel', '/tmp/regressor.t7', 'Save model to file')
cmd:option('-cuda', false, 'Use CUDA')
local opt = cmd:parse(arg)

-- try to use CUDA if required
if opt.cuda then
	require 'cunn'
	require 'cutorch'
end

-- logger
logger = optim.Logger(opt.logFile)

-- load training data
local fid = hdf5.open(opt.trainFile, 'r')
local X_trn = fid:read('X'):all():transpose(1,2)
local y_trn = fid:read('Y'):all():transpose(1,2)
y_trn = y_trn[{{}, opt.target}]
fid:close()

local N = X_trn:size(1) -- nr. of data points
local D = X_trn:size(2) -- nr. of dimensions

-- build regression model
local regressor = nn.Sequential()
regressor:add(nn.Linear(D,64))
regressor:add(nn.BatchNormalization(64))
regressor:add(nn.ReLU())
regressor:add(nn.Linear(64,1))
regressor:add(nn.ReLU())
if opt.cuda then
  regressor:cuda()
end

-- get paramters + gradients
local theta, gradTheta = regressor:getParameters()

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
  local x_hat = regressor:forward(x)
  -- compute loss
  local loss = criterion:forward(x_hat, y)
  -- gradient wrt loss
  local grad_loss = criterion:backward(x_hat, y)
  -- backpropagate
  regressor:backward(x, grad_loss)
  logger:add({'opfunc [loss]: '..loss})
  return loss, gradTheta
end

-- training
regressor:training()
for epoch = 1, opt.epochs do
  local indices = torch.randperm(N):long():split(opt.batchSize)
  indices[#indices] = nil

  -- run over minibatches
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)
    x = X_trn:index(1, v) -- batch data
    y = y_trn:index(1, v) -- batch data regression target
    if opt.cuda then
      x = x:cuda()
      y = y:cuda()
    end
    tmp, batch_loss = optim.adam(opfunc, theta, config)
  end
end

-- testing
if opt.test then
  -- load testing data
  local fid = hdf5.open(opt.testFile, 'r')
  local X_tst = fid:read('X'):all():transpose(1,2)
  local y_tst = fid:read('Y'):all():transpose(1,2)
	y_tst = y_tst[{{}, opt.target}]
  fid:close()

  if opt.cuda then
  	X_tst = X_tst:cuda()
  	y_tst = y_tst:cuda()
  end

  assert(D==X_tst:size(2), 'Whoops...')

  regressor:evaluate()
  y_hat = regressor:forward(X_tst)
  y_err = criterion:forward(y_hat, y_tst)
  print('MSE [m]: '..y_err)

  fid = hdf5.open(opt.evaluationFile, 'w')
  fid:write('/y_hat', y_hat:float())
  fid:write('/y_tst', y_tst:float())
  fid:close()
end

if opt.cuda then
	regressor = regressor:float()
end
torch.save(opt.saveModel, regressor)
