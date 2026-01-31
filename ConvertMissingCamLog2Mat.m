%% Setup and File Discovery
% Set root folder
root_folder = 'F:\OMR_Ontogeny_VOL';

% Verify folder exists before starting
if ~isfolder(root_folder)
    error('Root folder not found: %s', root_folder);
end

disp(['Scanning folder structure: ' root_folder]);
disp('This may take a moment...');

% Find files that start with "OMR_Ontogeny_VOL" and end with "000.txt"
file_pattern = fullfile(root_folder, '**', 'OMR_Ontogeny_VOL_*000.txt'); % Uses wildcard for subdirectories (**) 
files = dir(file_pattern);

%% Identify Missing .mat Files
disp('Checking which files are missing their .mat version...');

% Pre-allocate lists
missing_indices = false(length(files), 1);
mat_targets_list = cell(length(files), 1);

for i = 1:length(files)
    txt_full_path = fullfile(files(i).folder, files(i).name);
    
    % Define the target .mat file path
    % We replace the .txt extension with .mat
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
    return;
end

%% Create Tracking Table
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
        % --- LOAD DATA ---
        % Attempt to read the text file. 
        % Since we cannot rely on the custom 'txt2mat' function being present,
        % we use 'readtable' which is standard and robust for columns of data.
        % If your data has no headers, 'importdata' is a good fallback.
        a = readtable(current_txt_path);
        
        if isempty(a)
            a = importdata(current_txt_path);
        end
        
        % --- SAVE DATA ---
        % Save the variable 'a' into the .mat file
        % Using -v7.3 to support large files (>2GB) if necessary
        save(current_mat_target, 'a', '-v7.3');
        
        % --- UPDATE TABLE STATUS ---
        MissingFilesTable.Status{i} = 'Converted';
        fprintf('Done.\n');
        
    catch ME
        % Update table with error message
        MissingFilesTable.Status{i} = ['Error: ' ME.message];
        fprintf('FAILED.\n');
        warning('Failed to convert: %s', current_txt_path);
    end
end

disp('------------------------------------------------');
disp('Process Complete.');
disp('Check the "MissingFilesTable" variable in the workspace for results.');