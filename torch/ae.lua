-- autoencoder, author: rkwitt, mdixit (2016)
require 'optim'
require 'hdf5'
require 'nn'

-- cmdline parsing
local cmd = torch.CmdLine()
cmd:option('-logFile', '/tmp/ae.log', 'Logfile')
cmd:option('-dataFile', '/tmp/data.hdf5', 'HDF5 data file')
cmd:option('-learningRate', 0.001, 'Learning rate')
cmd:option('-epochs', 10, 'Training epochs')
cmd:option('-batchSize', 300, 'Batchsize')
local opt = cmd:parse(arg)

-- load some training/testing data
local fid = hdf5.open(opt.dataFile, 'r')
local src = fid:read('X_source_trn'):all() -- (training) object activations
local tst = fid:read('X_source_val'):all() -- (validation) object activations
local dst = fid:read('X_target'):all() -- (training) target
fid:close()

local N = src:size(1) -- nr. of data points
local D = src:size(2) -- nr. of dimensions
assert(D==dst:size(2), 'Whoops...')

print('#Source (training) data points: '..N..'x'..D)
print('#Source (validation) data points: '..tst:size(1)..'x'..tst:size(2))
print('#Target data points: '..dst:size(1)..'x'..dst:size(2))

-- logger
logger = optim.Logger(opt.logFile)

-- very simple encoder/decoder architecture
local p=0.5
local ae = nn.Sequential()
ae:add(nn.Linear(D,512))            -- ENC: dim -> 128
ae:add(nn.ReLU())                   -- ENC: ReLU
ae:add(nn.Dropout(p))             -- ENC: dropout

ae:add(nn.Linear(512,256))          -- ENC: 128 -> 64
ae:add(nn.ReLU())                   -- ENC: ReLU
ae:add(nn.Dropout(p))             -- ENC: dropout

ae:add(nn.Linear(256,64))          -- ENC: 128 -> 64
ae:add(nn.ReLU())                   -- ENC: ReLU
ae:add(nn.Dropout(p))             -- ENC: dropout

ae:add(nn.Linear(64,256))           -- DEC: 64 -> 128
ae:add(nn.ReLU())                   -- DEC: ReLU
ae:add(nn.Dropout(p))             -- DEC: dropout

ae:add(nn.Linear(256,512))          -- DEC: 64 -> 128
ae:add(nn.ReLU())                   -- DEC: ReLU
ae:add(nn.Dropout(p))             -- DEC: dropout

ae:add(nn.Linear(512,D))            -- DEC: 128 -> dim
ae:add(nn.ReLU())
print(ae)

-- configure optimizer
local config = { learningRate = opt.learningRate }
print(config)

-- model paramters
local theta, gradTheta = ae:getParameters()

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
  local x_hat = ae:forward(x)
  -- compute loss
  local loss = criterion:forward(x_hat, y)
  -- gradient wrt loss
  local grad_loss = criterion:backward(x_hat, y)
  -- backpropagate
  ae:backward(x, grad_loss)

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
    y = dst:index(1, v) -- batch dst
    tmp, batch_loss = optim.adam(opfunc, theta, config)
  end
end

-- TODO: should be done better (e.g., using -test as cmdline option)
out = ae:forward(tst)
local out_fid = hdf5.open('/Users/rkwitt/Remote/GuidedAugmentation/python/test/prediction.hdf5', 'w')
out_fid:write('/prediction', ae:forward(tst))
out_fid:close()
