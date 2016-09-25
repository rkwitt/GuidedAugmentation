require 'nn'
require 'optim'
require 'hdf5'

-- cmdline parsing
local cmd = torch.CmdLine()
cmd:option('-logFile', '/tmp/regressor.log', 'Logfile')
cmd:option('-trainFile', '/tmp/train.hdf5', '(Input) HDF5 training data file')
cmd:option('-testFile', '/tmp/test.hdf5', '(Input) HDF5 testing data file')
cmd:option('-predictionFile', '/tmp/prediction.hdf5', '(Output) HDF5 test data predictions')
cmd:option('-test', false, 'Run on test file')
cmd:option('-learningRate', 0.001, 'Learning rate')
cmd:option('-epochs', 10, 'Training epochs')
cmd:option('-batchSize', 300, 'Batchsize')
cmd:option('-saveModel', '/tmp/regressor.t7', 'Save model to file')
local opt = cmd:parse(arg)

-- logger
logger = optim.Logger(opt.logFile)

-- load training data
local fid = hdf5.open(opt.trainFile, 'r')
local X_trn = fid:read('X_trn'):all():transpose(1,2)
local y_trn = fid:read('y_trn'):all()
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

-- get paramters + gradients
local theta, gradTheta = regressor:getParameters()

-- MSE loss
local criterion = nn.MSECriterion()

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
    tmp, batch_loss = optim.adam(opfunc, theta, config)
  end
end

-- testing
if opt.test then
  -- load testing data
  local fid = hdf5.open(opt.testFile, 'r')
  local X_tst = fid:read('X_tst'):all():transpose(1,2)
  local y_tst = fid:read('y_tst'):all()
  fid:close()

  assert(D==X_tst:size(2), 'Whoops...')

  regressor:evaluate()
  y_hat = regressor:forward(X_tst)
  y_err = criterion:forward(y_hat, y_tst)
  print('MSE [m]: '..y_err)

  fid = hdf5.open('/tmp/debug.hdf5', 'w')
  fid:write('/y_hat', y_hat)
  fid:write('/y_tst', y_tst)
  fid:close()

  -- DEBUG
  -- for i=1,y_hat:size(1) do
  --   print(y_hat[i][1].."--"..y_tst[i])
  -- end
end

torch.save(opt.saveModel, regressor)
