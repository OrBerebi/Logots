% MATLAB script to read, process, and plot IMU data from a CSV file
% where yaw, pitch, and roll are stored as string arrays.
%
% This script generates a 3-panel plot showing the mean value and
% the min/max range for each sensor reading over time.

clearvars;
close all;
clc;

% --- 1. Configuration ---
filename = 'stg_imu_data.csv'; % Make sure this file is in your MATLAB path

% --- 2. Read Data ---
fprintf('Reading data from %s...\n', filename);

% Set import options to ensure yaw, pitch, and roll are read as strings
opts = detectImportOptions(filename);
if ~ismember('yaw', opts.VariableNames) || ~ismember('pitch', opts.VariableNames) || ~ismember('roll', opts.VariableNames)
    fprintf('Error: The CSV file must contain "yaw", "pitch", and "roll" columns.\n');
    return;
end

% Use setvartype for robust type assignment, replacing the direct assignment
try
    opts = setvartype(opts, {'yaw', 'pitch', 'roll'}, {'string', 'string', 'string'});
catch ME_settype
    fprintf('Warning: Could not set variable types using setvartype. Trying direct assignment. Error: %s\n', ME_settype.message);
    % Fallback to direct assignment just in case (e.g., very old MATLAB)
    opts.VariableTypes{'yaw'} = 'string';
    opts.VariableTypes{'pitch'} = 'string';
    opts.VariableTypes{'roll'} = 'string';
end

try
    data = readtable(filename, opts);
catch ME_read
    fprintf('Error reading file: %s\n', ME_read.message);
    fprintf('Please make sure the file "%s" is in the correct directory.\n', filename);
    return;
end

fprintf('Successfully read %d rows.\n', height(data));

% --- 3. Process Data ---
fprintf('Processing data... This may take a moment.\n');

numRows = height(data);

% Pre-allocate arrays for processed data
yaw_mean = zeros(numRows, 1);
yaw_min = zeros(numRows, 1);
yaw_max = zeros(numRows, 1);

pitch_mean = zeros(numRows, 1);
pitch_min = zeros(numRows, 1);
pitch_max = zeros(numRows, 1);

roll_mean = zeros(numRows, 1);
roll_min = zeros(numRows, 1);
roll_max = zeros(numRows, 1);

% Use frame_id as the primary x-axis
frames = data.frame_id;

% Loop through each row to parse the string arrays
for i = 1:numRows
    try
        % Parse Yaw: jsondecode is a safe way to handle "[...]" strings
        yaw_data = jsondecode(data.yaw(i));
        % Handle case where jsondecode returns an empty array
        if isempty(yaw_data)
            yaw_mean(i) = NaN; yaw_min(i) = NaN; yaw_max(i) = NaN;
        else
            yaw_mean(i) = mean(yaw_data);
            yaw_min(i) = min(yaw_data);
            yaw_max(i) = max(yaw_data);
        end
        
        % Parse Pitch
        pitch_data = jsondecode(data.pitch(i));
        if isempty(pitch_data)
            pitch_mean(i) = NaN; pitch_min(i) = NaN; pitch_max(i) = NaN;
        else
            pitch_mean(i) = mean(pitch_data);
            pitch_min(i) = min(pitch_data);
            pitch_max(i) = max(pitch_data);
        end
        
        % Parse Roll
        roll_data = jsondecode(data.roll(i));
        if isempty(roll_data)
            roll_mean(i) = NaN; roll_min(i) = NaN; roll_max(i) = NaN;
        else
            roll_mean(i) = mean(roll_data);
            roll_min(i) = min(roll_data);
            roll_max(i) = max(roll_data);
        end
        
    catch ME
        fprintf('Warning: Could not parse row %d. Skipping. Error: %s\n', i, ME.message);
        % Assign NaN to skipped rows so they don't plot badly
        yaw_mean(i) = NaN; yaw_min(i) = NaN; yaw_max(i) = NaN;
        pitch_mean(i) = NaN; pitch_min(i) = NaN; pitch_max(i) = NaN;
        roll_mean(i) = NaN; roll_min(i) = NaN; roll_max(i) = NaN;
    end
end

fprintf('Data processing complete.\n');

% --- 4. Plot Visualization ---
fprintf('Generating plots...\n');

fig = figure('Name', 'IMU Sensor Data (Yaw, Pitch, Roll)', 'NumberTitle', 'off', 'WindowState', 'maximized');

% Create the x-axis coordinates for the shaded patch
% This creates a polygon that goes out along the min and back along the max
x_axis = [frames; flipud(frames)];

% --- Yaw Plot ---
ax1 = subplot(3, 1, 1);
hold on;
% Create the y-axis coordinates for the shaded patch
yaw_fill = [yaw_min; flipud(yaw_max)];
% Plot the shaded region
patch(x_axis, yaw_fill, 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Min/Max Range');
% Plot the mean line
plot(frames, yaw_mean, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Mean Yaw');
ylim([0, 360]);
title('Yaw over Time');
xlabel('Frame ID');
ylabel('Yaw (degrees)');
legend('Location', 'best');
grid on;
box on;
hold off;

% --- Pitch Plot ---
ax2 = subplot(3, 1, 2);
hold on;
pitch_fill = [pitch_min; flipud(pitch_max)];
patch(x_axis, pitch_fill, 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Min/Max Range');
plot(frames, pitch_mean, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Mean Pitch');
ylim([-45, 45]);
title('Pitch over Time');
xlabel('Frame ID');
ylabel('Pitch (degrees)');
legend('Location', 'best');
grid on;
box on;
hold off;

% --- Roll Plot ---
ax3 = subplot(3, 1, 3);
hold on;
roll_fill = [roll_min; flipud(roll_max)];
patch(x_axis, roll_fill, 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Min/Max Range');
plot(frames, roll_mean, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Mean Roll');
ylim([-50, 50]);
title('Roll over Time');
xlabel('Frame ID');
ylabel('Roll (degrees)');
legend('Location', 'best');
grid on;
box on;
hold off;

% Link axes for synchronized zooming and panning
linkaxes([ax1, ax2, ax3], 'x');

% Add a main title to the figure
sgtitle('IMU Sensor Output Analysis (Mean with Min/Max Range)');

fprintf('Done. Plot window is active.\n');

