function config = AGA_setup(sysType)

config = [];
if nargin < 1
    disp('MacBook_rkwitt');
    disp('eisbaer');
    disp('grassmann');
    disp('polarbaer');
    return;
end


% Add Selective Search paths
addpath(fullfile(pwd, 'selective_search'));
addpath(fullfile(pwd, 'selective_search/Dependencies'));
addpath(fullfile(pwd, 'selective_search/Dependencies/anigaussm/'));
addpath(fullfile(pwd, 'selective_search/Dependencies/FelzenSegment/'));

% Load system-specific configuration
if strcmp(sysType, 'MacBook_rkwitt') % rkwitt laptop

    config = load('config/config_MacBook_rkwitt');
    eval('config = config.config_MacBook_rkwitt');
end
if strcmp(sysType, 'eisbaer') % GPU server

  	config = load('config/config_eisbaer');
		eval('config = config.config_eisbaer');

end
if strcmp(sysType, 'polarbaer') % GPU server

  	config = load('config/config_polarbaer');
		eval('config = config.config_polarbaer');

end
if strcmp(sysType, 'grassmann') % rkwitt iMac

  	config = load('config/config_grassmann');
		eval('config = config.config_grassmann');

end