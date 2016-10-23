function config = ga_setup(sysType)

% Add Selective Search paths
addpath(fullfile(pwd, 'selective_search'));
addpath(fullfile(pwd, 'selective_search/Dependencies'));
addpath(fullfile(pwd, 'selective_search/Dependencies/anigaussm/'));
addpath(fullfile(pwd, 'selective_search/Dependencies/FelzenSegment/'));

% Load system-specific configuration
if strcmp(sysType, 'MacBook_rkwitt')
    
    config = load('config_MacBook_rkwitt');
    eval('config = config.config_MacBook_rkwitt');
    
end