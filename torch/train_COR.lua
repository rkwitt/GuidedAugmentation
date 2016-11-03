-- train covariate regressor (COR) on data
require 'nn'
require 'optim'
require 'hdf5'

-- cmdline parsing
local cmd = torch.CmdLine()
cmd:option('-logFile',        '/tmp/train_COR.log',   'Logfile')
cmd:option('-dataFile',       '/tmp/data.hdf5',       'HDF5 input data')
cmd:option('-saveCOR',        '/tmp/model_COR.t7',    'Save model to file')
cmd:option('-column',         1,                      'Column of covariate data [1=Depth, 2=Pose]')
cmd:option('-learningRate',   0.001,                  'Learning rate')
cmd:option('-epochs',         10,                     '#Training epochs')
cmd:option('-batchSize',      300,                    'Batchsize')
cmd:option('-cuda',           false,                  'Use CUDA')
local opt = cmd:parse(arg)

-- try to use CUDA if required
if opt.cuda then
  require 'cunn'
  require 'cutorch'
end

local fid = hdf5.open(opt.dataFile, 'r')
local X = fid:read('X'):all():transpose(1,2)
local Y = fid:read('Y'):all():transpose(1,2)
local Y = Y[{{}, opt.column}] -- get column of target covariate values
fid:close()

local N = X:size(1) -- #data points
local D = X:size(2) -- #dimensions

-- logger
logger = optim.Logger(opt.logFile)

--create regressor
regressor = nn.Sequential()
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

regressor:training()
for epoch = 1, opt.epochs do

    local indices = torch.randperm(N):long():split(opt.batchSize)
    indices[#indices] = nil

    -- run over minibatches
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)
        x = X:index(1, v) -- batch data
        y = Y:index(1, v) -- batch data regression target
        if opt.cuda then
            x = x:cuda()
            y = y:cuda()
        end
        tmp, batch_loss = optim.adam(opfunc, theta, config)
    end
end

--make sure we can save the model
if opt.cuda then
    regressor = regressor:float()
end
torch.save(opt.saveCOR, regressor)