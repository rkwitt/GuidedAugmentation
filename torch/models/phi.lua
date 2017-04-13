--Network implementing phi

require 'nn'


local p=0.25 -- dropout probability
local D=4096
local A=256
local B=32

local phi = nn.Sequential()
phi:add(nn.Linear(D,A))
phi:add(nn.BatchNormalization(A))
phi:add(nn.ELU())
phi:add(nn.Dropout(p))

phi:add(nn.Linear(A,B))
phi:add(nn.BatchNormalization(B))
phi:add(nn.ELU())
phi:add(nn.Dropout(p))

phi:add(nn.Linear(B,A))
phi:add(nn.BatchNormalization(A))
phi:add(nn.ELU())
phi:add(nn.Dropout(p))

phi:add(nn.Linear(A,D))
phi:add(nn.ReLU())

return phi
