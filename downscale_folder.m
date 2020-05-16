function f = downscale_folder(folder, output_folder, downscale_factor, new_ext)
    if ~exist(output_folder, 'dir')
        mkdir(output_folder)
    end
    imagefiles = dir(folder);
    for ii=1:length(imagefiles)
        if ii <= 2
            continue
        end
        currentfilename = imagefiles(ii).name;
        [~, name, ~] = fileparts(currentfilename);
        currentimage = imread(fullfile(folder, currentfilename));
        width = floor(size(currentimage, 1) / downscale_factor);
        height = floor(size(currentimage, 2) / downscale_factor);
        J = imresize(currentimage, [width, height], 'bicubic');
        new_name = fullfile(output_folder, strcat(name, '.', new_ext));
        imwrite(J, new_name);
    end
    f = 1;
end