require 'nn'


local regressor = nn.Sequential()
regressor:add(nn.Linear(4096,64))
regressor:add(nn.BatchNormalization(64))regressor:add(nn.ReLU())
regressor:add(nn.Linear(64,1))
regressor:add(nn.ReLU())

return regressor
