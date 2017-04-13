--train encoder-decoder network implementing phi

require 'optim'
require 'hdf5'
require 'nn'


-- cmdline parsing
local cmd = torch.CmdLine()
-- input/output files
cmd:option('-dataFile',             '/tmp/data.hdf5',           'HDF5 data file')
cmd:option('-saveModel',            '/tmp/model.t7',            'Save trained phi model to file')
cmd:option('-logFile',              '/tmp/train_phi.log',       'Logfile')
cmd:option('-target',               4,                          'Covariate target value')

-- input models
cmd:option('-modelPhi',              '/tmp/pretrained_phi.t7',   'Pretrained encoder-decoder network (gamma)')
cmd:option('-modelGamma',            '/tmp/pretrained_gamma.t7', 'Pretrained attribute regressor (phi)')

-- misc. options
cmd:option('-learningRate',         0.001,                      'Learning rate')
cmd:option('-epochs',               10,                         'Training epochs')
cmd:option('-batchSize',            300,                        'Batchsize')
cmd:option('-cuda',                 false,                      'Use CUDA')

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
local src = fid:read('X'):all():transpose(1,2)
fid:close()

local N = src:size(1) -- nr. of data points
local D = src:size(2) -- nr. of dimensions
print('#Source data points: '..N..'x'..D)

-- load pretrained gamma and freeze layers (these are not trained)
local modelGamma = torch.load(opt.modelGamma)
for i, m in ipairs(modelGamma.modules) do
    m.accGradParameters = function() end
    m.updateParameters  = function() end
end
print('Froze gamma layers')

-- load pretrained phi
print( opt.modelPhi )
modelPhi = torch.load(opt.modelPhi)

-- split
local modelSNN = nn.ConcatTable()
modelSNN:add(nn.Identity())
modelSNN:add(modelGamma)

-- stack model together
model = nn.Sequential()
model:add(modelPhi) 
model:add(modelSNN) 

if opt.cuda then
    model:cuda()
end

-- configure optimizer
local config = { learningRate = opt.learningRate }
print(config)

-- model paramters
local theta, gradTheta = model:getParameters()

-- MSE loss
local criterion = nn.ParallelCriterion()
criterion:add(nn.MSECriterion(),0.7) --MSE loss (regularizer)
criterion:add(nn.MSECriterion(),0.3) --MSE loss (mismatch penalty)

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
    indices[#indices] = nil -- set last batch to nil
    
    -- run over minibatches
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)
        x = src:index(1, v) -- batch src
        y = torch.Tensor(x:size(1),1):fill(opt.target)
        if opt.cuda then
            x = x:cuda()
            y = {x, y:cuda()}
        end
        tmp, batch_loss = optim.adam(opfunc, theta, config)
    end
end

--make sure we can save the trained phi model
if opt.cuda then
    model = model:float()
end
torch.save( opt.saveModel, model)