%%% Code writen by Pedro Tomas 2026 %%%
%%% This script finds camlog and stimlog files from ONTOGENY OMR experiment
%%% copies them and saves them in a new folder.

clear;

%% 1. Path Configuration
root_folder = 'F:\OMR_Ontogeny_VOL';
cam_dest = fullfile(root_folder, 'Camlog_Mat_Files');
stim_dest = fullfile(root_folder, 'Stimlog_Mat_Files');

% Initialize output directories
if ~isfolder(cam_dest), mkdir(cam_dest); end
if ~isfolder(stim_dest), mkdir(stim_dest); end

%% 2. Find files (recursive)
% Recursively find all .mat files
mat_pattern = fullfile(root_folder, '**', '*.mat');
all_mat_files = dir(mat_pattern);

% Exclude files already located in the destination folders
exclude_idx = contains({all_mat_files.folder}, {'Camlog_Mat_Files', 'Stimlog_Mat_Files'});
files_to_process = all_mat_files(~exclude_idx);

%% 3. Migration Logic
cam_count = 0;
stim_count = 0;

for i = 1:length(files_to_process)
    source = fullfile(files_to_process(i).folder, files_to_process(i).name);
    fname = files_to_process(i).name;
    
    % Identify file type and set target
    if contains(fname, 'OMR_Ontogeny_VOL') && contains(fname, '000.mat')
        target = fullfile(cam_dest, fname);
        [status, ~] = copyfile(source, target);
        if status, cam_count = cam_count + 1; end
        
    elseif contains(fname, 'stimlog_') && contains(fname, '.mat')
        target = fullfile(stim_dest, fname);
        [status, ~] = copyfile(source, target);
        if status, stim_count = stim_count + 1; end
    end
end

%% 4. Final Audit
fprintf('------------------------------------------------\n');
fprintf('Migration Summary:\n');
fprintf('Camlog .mat files copied: %d\n', cam_count);
fprintf('Stimlog .mat files copied: %d\n', stim_count);
fprintf('------------------------------------------------\n');