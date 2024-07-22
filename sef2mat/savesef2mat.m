% Step 1: Read the data and header information of the SEF file
[data, hdr] = readsef('example.sef');

% Step 2: Save the data as a .mat file
% Create a structure to hold the data
output_data = struct();
output_data.data = data.data; % EEG data
output_data.srate = data.srate; % Sampling Rate
output_data.sensor_labels = data.sensor_labels; % Sensor Tags
output_data.filename = data.filename; % File name
output_data.header = hdr; % Header Information

% Save as .mat file
save('example.mat', '-struct', 'output_data');
