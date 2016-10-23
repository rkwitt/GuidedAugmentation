require 'nn'

local p=0.25
local ae = nn.Sequential()
ae:add(nn.Linear(4096, 256))
ae:add(nn.BatchNormalization(256))
ae:add(nn.ELU())
ae:add(nn.Dropout(p))

ae:add(nn.Linear(256,32))
ae:add(nn.BatchNormalization(32))
ae:add(nn.ELU())
ae:add(nn.Dropout(p))

ae:add(nn.Linear(32,256))
ae:add(nn.BatchNormalization(256))
ae:add(nn.ELU())
ae:add(nn.Dropout(p))

ae:add(nn.Linear(256, 4096))
ae:add(nn.ReLU())

return ae
