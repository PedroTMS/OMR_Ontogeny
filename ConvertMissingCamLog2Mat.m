%%% Code writen by Pedro Tomás 31/01/2026
%%% Designed to automate the conversion of raw camera log text files
%%% into .mat files while avoiding redundancy; recursively scans a
%%% given directory for raw .txt files, checks if a converted .mat
%%% file already exists, makes a table of only those without, and
%%% iterates the ones without to convert the .txt to .mat files

clear

%% Setup main path
% Set root folder
root_folder = 'F:\OMR_Ontogeny_VOL';

% Verify folder exists before starting
if ~isfolder(root_folder)
    error('Root folder not found: %s', root_folder);
end

%% File Discovery
disp(['Scanning folder structure: ' root_folder]);
disp('This may take a moment...');

% Find cam_log files
% Find files that start with "OMR_Ontogeny_VOL" and end with "000.txt"
cam_pattern = fullfile(root_folder, '**', 'OMR_Ontogeny_VOL_*000.txt'); % Uses wildcard for subdirectories (**) and wildcard(*)for file ending
cam_files = dir(cam_pattern);

% Find Stimulus Log files
% Find files starting with "stimlog_" and ending with ".txt"
stim_pattern = fullfile(root_folder, '**', 'stimlog_*.txt'); % Uses wildcard (*) for file ending
stim_files = dir(stim_pattern);

% Concatenate both structure arrays into one
all_files = [cam_files; stim_files];

%% CHECKPOINT: Remove Duplicate Files
% Filter files by the unique full file path.

if ~isempty(all_files)
    % Create a cell array of full paths for every file found
    full_paths = fullfile({all_files.folder}, {all_files.name});
    
    % Find unique paths and their indices
    [~, unique_idx, ~] = unique(full_paths);
    
    % Calculate how many duplicates were found
    num_duplicates = length(all_files) - length(unique_idx);
    
    if num_duplicates > 0
        disp(['Duplicate Check: Found and removed ' num2str(num_duplicates) ' duplicate files.']);
        % Keep only the unique files
        all_files = all_files(unique_idx);
    else
        disp('Duplicate Check: No duplicates found.');
    end
else
    disp('No files found matching the search criteria.');
    return; % froce stop the script
end

%% Identify Missing .mat Files
disp(['Found ' num2str(length(all_files)) ' total candidate text files.']);
disp('Checking which files are missing their .mat version...');

% Pre-allocate lists
missing_indices = false(length(files), 1);
mat_targets_list = cell(length(files), 1);

% Iterate all files to check if the .mat file exists
for i = 1:length(files)
    txt_full_path = fullfile(files(i).folder, files(i).name);
    
    % Define the target .mat file path, replace the .txt extension with .mat
    mat_full_path = regexprep(txt_full_path, '\.txt$', '.mat');
    
    % Store the target path for later use
    mat_targets_list{i} = mat_full_path;
    
    % Check if the .mat file already exists
    if ~isfile(mat_full_path)
        missing_indices(i) = true;
    end
end

% Filter to only the missing files
files_to_convert = files(missing_indices);
targets_to_create = mat_targets_list(missing_indices);

if isempty(files_to_convert)
    disp('No files found that need conversion.');
    return; % force stop the script
end

%% Create Info Table
% Create table with file details and a Status column
FileNames = {files_to_convert.name}';
Folders = {files_to_convert.folder}';
TargetMatFiles = targets_to_create;
Status = repmat({'Pending'}, length(files_to_convert), 1);

MissingFilesTable = table(FileNames, Folders, TargetMatFiles, Status);

disp('------------------------------------------------');
fprintf('Found %d files missing .mat versions.\n', height(MissingFilesTable));
disp('Starting conversion...');
disp('------------------------------------------------');

%% Process Files Fish by Fish
% Iterate through the table and convert
for i = 1:height(MissingFilesTable)
    
    current_txt_path = fullfile(MissingFilesTable.Folders{i}, MissingFilesTable.FileNames{i});
    current_mat_target = MissingFilesTable.TargetMatFiles{i};
    
    fprintf('Processing %d/%d: %s ... ', i, height(MissingFilesTable), MissingFilesTable.FileNames{i});
    
    try
        % LOAD DATA
        % Call txt2mat function
        a = txt2mat(current_txt_path);
        
        % SAVE DATA
        % Save 'a' into the .mat file
        save(current_mat_target, 'a', '-v7.3');
        
        % UPDATE TABLE STATUS
        MissingFilesTable.Status{i} = 'Converted';
        fprintf('Done.\n');
        
    catch ME % MATLAB Exception
        % Update table with error message if txt2mat fails
        MissingFilesTable.Status{i} = ['Error: ' ME.message]; % message -> automatically generated
        fprintf('FAILED.\n');
        warning('Failed to convert: %s\nError message: %s', current_txt_path, ME.message);
    end
end

disp('------------------------------------------------');
disp('__!!!ALL DONE!!!__');
disp('Check the "MissingFilesTable" for results.');


% %% Process Files Fish by Fish
% % Iterate through the table and convert
% for i = 1:height(MissingFilesTable)
%     
%     current_txt_path = fullfile(MissingFilesTable.Folders{i}, MissingFilesTable.FileNames{i});
%     current_mat_target = MissingFilesTable.TargetMatFiles{i};
%     
%     fprintf('Processing %d/%d: %s ... ', i, height(MissingFilesTable), MissingFilesTable.FileNames{i});
%     
%     try
%         % --- LOAD DATA ---
%         % Attempt to read the text file. 
%         % Since we cannot rely on the custom 'txt2mat' function being present,
%         % we use 'readtable' which is standard and robust for columns of data.
%         % If your data has no headers, 'importdata' is a good fallback.
%         a = readtable(current_txt_path);
%         
%         if isempty(a)
%             a = importdata(current_txt_path);
%         end
%         
%         % --- SAVE DATA ---
%         % Save the variable 'a' into the .mat file
%         % Using -v7.3 to support large files (>2GB) if necessary
%         save(current_mat_target, 'a', '-v7.3');
%         
%         % --- UPDATE TABLE STATUS ---
%         MissingFilesTable.Status{i} = 'Converted';
%         fprintf('Done.\n');
%         
%     catch ME
%         % Update table with error message
%         MissingFilesTable.Status{i} = ['Error: ' ME.message];
%         fprintf('FAILED.\n');
%         warning('Failed to convert: %s', current_txt_path);
%     end
% end
% 
% disp('------------------------------------------------');
% disp('Process Complete.');
% disp('Check the "MissingFilesTable" variable in the workspace for results.');