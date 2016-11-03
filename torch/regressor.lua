require 'nn'
require 'optim'
require 'hdf5'

-- cmdline parsing
local cmd = torch.CmdLine()
cmd:option('-logFile',        '/tmp/regressor.log',   'Logfile')
cmd:option('-predictionFile', '/tmp/evaluation.hdf5', 'File with predictions + ground truth')
cmd:option('-trainFile',      '/tmp/train.hdf5',      '(Input) HDF5 training data file')
cmd:option('-testFile',       '/tmp/test.hdf5',       '(Input) HDF5 testing data file')
cmd:option('-test',           false,                  'Run regressor on test file')
cmd:option('-train',          false,                  'Train regressor')
cmd:option('-target',         1,                      '1=Depth, 2=Pose')
cmd:option('-learningRate',   0.001,                  'Learning rate')
cmd:option('-epochs',         10,                     '#Training epochs')
cmd:option('-batchSize',      300,                    'Batchsize')
cmd:option('-saveModel',      '',                     'Save model to file')
cmd:option('-loadModel',      '',                     'Load trained model')
cmd:option('-cuda',           false,                  'Use CUDA')
local opt = cmd:parse(arg)

-- try to use CUDA if required
if opt.cuda then
  require 'cunn'
  require 'cutorch'
end

-- logger
logger = optim.Logger(opt.logFile)

--load regressor if requested
local regressor = nil
if not(opt.loadModel == '') then
  regressor = torch.load( opt.loadModel )
else
  --create regressor
  regressor = nn.Sequential()
  regressor:add(nn.Linear(D,64))
  regressor:add(nn.BatchNormalization(64))
  regressor:add(nn.ReLU())
  regressor:add(nn.Linear(64,1))
  regressor:add(nn.ReLU())
end

local fid   = nil
local X_trn = nil
local y_trn = nil
local N     = nil
local D     = nil

-- load training data
if opt.train then
  fid = hdf5.open(opt.trainFile, 'r')
  X_trn = fid:read('X_trn'):all():transpose(1,2)
  y_trn = fid:read('Y_trn'):all():transpose(1,2)
  y_trn = y_trn[{{}, opt.target}]
  fid:close()

  N = X_trn:size(1) -- #data points
  D = X_trn:size(2) -- #dimensions
end

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
if opt.train then
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
end

-- testing
if opt.test then
  -- load testing data
  local fid = hdf5.open(opt.testFile, 'r')
 
  local X_tst = fid:read('X_tst'):all()
  local y_tst = fid:read('Y_tst'):all()

  --case 1: we only have one input vector
  if (X_tst:size():size() == 1) then
    X_tst = X_tst:reshape(1, X_tst:size(1))
    y_tst = y_tst:reshape(1, y_tst:size(1))
  else
  -- case 2: we have an array of input vectors
    X_tst = X_tst:transpose(1,2)
    y_tst = y_tst:transpose(1,2)
  end
  y_tst = y_tst[{{}, opt.target}]
  fid:close()

  if opt.cuda then
    X_tst = X_tst:cuda()
    y_tst = y_tst:cuda()
  end

  regressor:evaluate()
  y_hat = regressor:forward(X_tst)
  y_err = criterion:forward(y_hat, y_tst)  
  --print('MSE [m]: '..y_err)
  
  fid = hdf5.open(opt.predictionFile, 'w')
  fid:write('/y_hat', y_hat:float())
  fid:write('/y_tst', y_tst:float())
  fid:close()
end

if opt.cuda then
  regressor = regressor:float()
end
if not(opt.saveModel == '') then
  torch.save(opt.saveModel, regressor)
end