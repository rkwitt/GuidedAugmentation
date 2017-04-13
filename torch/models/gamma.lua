--Network implementing gamma

require 'nn'

local D = 4096
local A = 64

local gamma = nn.Sequential()
gamma:add(nn.Linear(D,A))
gamma:add(nn.BatchNormalization(A))
gamma:add(nn.ReLU())
gamma:add(nn.Linear(A,1))
gamma:add(nn.ReLU())

return gamma
