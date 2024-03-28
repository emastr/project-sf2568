% Load the matrix consisting of different colors (replace 'color_matrix.txt' with the path to your file)
color_matrix = dlmread('color.txt');

% Define a custom colormap with 256 colors
custom_colormap = rand(256, 3); % Generates a random colormap with 256 colors
%custom_colormap = parula(256);
%custom_colormap = colorcube(256);

% Plot the color matrix using imagesc and set the colormap
imagesc(color_matrix);
colormap(custom_colormap); % Set the custom colormap

% Add color bar to the plot
%colorbar;

% Set axis labels
%xlabel('Column');
%ylabel('Row');

% Add title
title('Visualization of Color Matrix');

% Display plot
axis equal;
axis tight;