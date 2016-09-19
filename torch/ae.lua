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
local src = fid:read('X_src_trn'):all() -- object activations
local dst = fid:read('X_res_trn'):all() -- object activations closest to src
local tst = fid:read('X_tst'):all() -- testing object activations
-- local src = fid:read('/src'):all():transpose(1,2)
-- local dst = fid:read('/dst'):all():transpose(1,2)

local N = src:size(1) -- nr. of data points
local D = src:size(2) -- nr. of dimensions
assert(D==dst:size(2), 'Whoops...')

print('#Source data points: '..N..'x'..D)
print('#Target data points: '..dst:size(1)..'x'..dst:size(2))

-- logger
logger = optim.Logger(opt.logFile)

-- very simple encoder/decoder architecture
local ae = nn.Sequential()
ae:add(nn.Linear(D,128))    -- ENC: dim -> 128
ae:add(nn.Tanh())           -- ENC: tanh non-linearity
ae:add(nn.Dropout(0.2))     -- ENC: dropout
ae:add(nn.Linear(128,64))   -- ENC: 128 -> 64
ae:add(nn.Tanh())           -- ENC: tanh non-linearity
ae:add(nn.Linear(64,128))   -- DEC: 64 -> 128
ae:add(nn.Tanh())           -- DEC: tanh non-linearity
ae:add(nn.Dropout(0.2))     -- DEC: dropout
ae:add(nn.Linear(128,D))    -- DEC: 128 -> dim
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

out = ae:forward(tst)
local out_fid = hdf5.open('/tmp/prediction.hdf5', 'w')
out_fid:write('/prediction', ae:forward(tst))
out_fid:close()
