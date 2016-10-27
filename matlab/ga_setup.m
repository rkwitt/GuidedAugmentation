function config = ga_setup(sysType)

config = [];
if nargin < 1
    disp('MacBook_rkwitt');
    disp('eisbaer');
    disp('grassmann');
    return;
end


% Add Selective Search paths
addpath(fullfile(pwd, 'selective_search'));
addpath(fullfile(pwd, 'selective_search/Dependencies'));
addpath(fullfile(pwd, 'selective_search/Dependencies/anigaussm/'));
addpath(fullfile(pwd, 'selective_search/Dependencies/FelzenSegment/'));

% Load system-specific configuration
if strcmp(sysType, 'MacBook_rkwitt') % rkwitt laptop

    config = load('config_MacBook_rkwitt');
    eval('config = config.config_MacBook_rkwitt');
end
if strcmp(sysType, 'eisbaer') % GPU server

  	config = load('config_eisbaer');
		eval('config = config.config_eisbaer');

end
if strcmp(sysType, 'grassmann') % rkwitt iMac

  	config = load('config_grassmann');
		eval('config = config.config_grassmann');

end
