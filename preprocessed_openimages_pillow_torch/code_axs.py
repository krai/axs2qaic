import errno
import os
import json
import sys

import numpy as np

import torch
import torchvision

from PIL import Image
def generate_file_list(supported_extensions, source_dir, calibration_dir=None, index_file=None, first_n=None, first_n_insert=None, images_directory=None):
    original_file_list = os.listdir(source_dir)
    sorted_filenames = [filename for filename in sorted(original_file_list) if any(filename.lower().endswith(extension) for extension in supported_extensions) ]
    if index_file:
        index_file = os.path.join(calibration_dir, index_file)
        with open(index_file, 'r') as file_in:
            sorted_filenames = []
            for line in file_in:
                sorted_filenames.append((line.split())[0])
                sorted_filenames.sort()
            first_n_insert = f'{index_file}_'
    elif first_n:
        sorted_filenames = sorted_filenames[:first_n] #if first_n is not None else sorted_filenames
        
        assert len(sorted_filenames) == first_n

    return sorted_filenames

 # Load and preprocess image
def load_image(image_path,                # Full path to processing image
               target_size,               # Desired size of resulting image
               data_type = 'uint8',       # Data type to store
               convert_to_bgr = False,    # Swap image channel RGB -> BGR
               normalize_symmetric = False,    # Normalize the data
               normalize_lower = -1,      # Normalize - lower limit
               normalize_upper = 1,       # Normalize - upper limit
               subtract_mean = False,     # Subtract mean
               given_channel_means = '',  # Given channel means
               given_channel_stds = '',
               quantized = False,          # Quantization type, True for quantized to int8 
               quant_scale = 1,           # Quantization scale 
               quant_offset = 0,          # Quantization offset 
               convert_to_unsigned  = False,   # True to convert from int to uint
               convert_dtype_before_resize = False # True to do the data type conversion before image resize
              ):

    img = Image.open(image_path).convert('RGB')

    img = torchvision.transforms.functional.to_tensor(img)

    mean = torch.as_tensor(given_channel_means)
    std = torch.as_tensor(given_channel_stds)

    img = (img - mean[:, None, None]) / std[:, None, None]

    img = torch.nn.functional.interpolate(img[None], size=(target_size, target_size), scale_factor=None, mode='bilinear',
                                            recompute_scale_factor=None, align_corners=False)[0]
    img = img.numpy()

    # Value 1 for quantization to uint8
    if quantized:
        img = quantized_to_uint8(img, quant_scale, quant_offset)

    # Value 1 to convert from int8 to uint8
    if convert_to_unsigned:
        img = int8_to_uint8(img)

    original_height, original_width, _ = img.shape
    # Make batch from single image
    batch_shape = (1, target_size, target_size, 3)
    batch_data = img.reshape(batch_shape)

    return batch_data, original_width, original_height


def quantized_to_uint8(image, scale, offset):
  image = image.astype(np.float64)
  quant_image = (image/scale + offset).astype(np.float64)
  output = np.round_(quant_image)
  output = np.clip(output, 0, 255)
  return output.astype(np.uint8)

def int8_to_uint8(image):
    image = (image+128).astype(np.uint8)
    return image


def preprocess_files(selected_filenames,
                     source_dir,
                     destination_dir,
                     resolution,
                     data_type,
                     convert_to_bgr,
                     normalize_symmetric,
                     normalize_lower,
                     normalize_upper,
                     subtract_mean,
                     given_channel_means,
                     given_channel_stds,
                     quantized,
                     quant_scale, 
                     quant_offset, 
                     convert_to_unsigned,
                     convert_dtype_before_resize,
                     new_file_extension):

    "Go through the selected_filenames and preprocess all the files"

    output_signatures = []

    for current_idx in range(len(selected_filenames)):
        input_filename = selected_filenames[current_idx]

        full_input_path     = os.path.join(source_dir, input_filename)

        image_data, original_width, original_height = load_image(image_path = full_input_path,
                                                                 target_size = resolution,
                                                                 data_type = data_type,
                                                                 convert_to_bgr = convert_to_bgr,
                                                                 normalize_symmetric = normalize_symmetric,
                                                                 normalize_lower = normalize_lower,
                                                                 normalize_upper = normalize_upper,
                                                                 subtract_mean = subtract_mean,
                                                                 given_channel_means = given_channel_means,
                                                                 given_channel_stds = given_channel_stds,
                                                                 quantized = quantized,
                                                                 quant_scale = quant_scale, 
                                                                 quant_offset = quant_offset, 
                                                                 convert_to_unsigned = convert_to_unsigned,
                                                                 convert_dtype_before_resize = convert_dtype_before_resize)

        output_filename = input_filename.rsplit('.', 1)[0] + '.' + new_file_extension if new_file_extension else input_filename

        full_output_path    = os.path.join(destination_dir, output_filename)
        image_data.tofile(full_output_path)

        print("[{}]:  Stored {}".format(current_idx+1, full_output_path) )

        output_signatures.append('{};{};{}'.format(output_filename, original_width, original_height) )
    return output_signatures


#intermediate_data_type  = np.int8       # affects the accuracy a bit

def preprocess(source_dir, annotations_filepath, calibration, resolution, offset, volume_str, fof_name, first_n, data_type, new_file_extension, image_file, data_layout, convert_to_bgr, input_data_type, normalize_symmetric, normalize_lower, normalize_upper, subtract_mean, given_channel_means, given_channel_stds, quant_scale, quant_offset, quantized, convert_to_unsigned, convert_dtype_before_resize, supported_extensions,file_name, normalayout, input_file_list, index_file=None, tags=None, entry_name=None, __record_entry__=None):
    
    intermediate_data_type  = np.float32
    __record_entry__["tags"] = tags or [ "preprocessed" ]
    if quantized:
        entry_end = "_quantized"
    elif calibration:
        entry_end = "_calibration"
    else:
        entry_end = ""    
    if not entry_name:
        first_n_insert = f'_first.{first_n}_images' if first_n and first_n != 50000 else 'full'
    entry_name = f'openimages_preprocessed_using_pillow_torch_{first_n_insert}{entry_end}'
    
    __record_entry__.save( entry_name )
    output_directory     = __record_entry__.get_path(file_name)
    os.makedirs( output_directory )
    destination_dir = output_directory

    if given_channel_means:
        given_channel_means = np.fromstring(given_channel_means, dtype=np.float32, sep=' ').astype(intermediate_data_type)
        if convert_to_bgr:
            given_channel_means = given_channel_means[::-1]     # swapping Red and Blue colour channels

    if given_channel_stds:
        given_channel_stds = np.fromstring(given_channel_stds, dtype=np.float32, sep=' ').astype(intermediate_data_type)
        if convert_to_bgr:
            given_channel_stds  = given_channel_stds[::-1]      # swapping Red and Blue colour channels


    print("From: {} , To: {} , Size: {} ,  OFF: {}, VOL: '{}', FOF: {}, DTYPE: {}, EXT: {}, IMG: {}".format(
        source_dir, destination_dir, resolution, offset, volume_str, fof_name, data_type, new_file_extension, image_file) )


    if image_file:
        source_dir          = os.path.dirname(image_file)
        selected_filenames  = [ os.path.basename(image_file) ]

    else:
        if annotations_filepath and not calibration:  # get the "coco-natural" filename order (not necessarily alphabetic)
            with open(annotations_filepath, "r") as annotations_fh:
                annotations_struct = json.load(annotations_fh)

            ordered_filenames = [ image_entry['file_name'] for image_entry in annotations_struct['images'] ]

        elif os.path.isdir(source_dir):     # in the absence of "coco-natural", use alphabetic order

            ordered_filenames = input_file_list #[filename for filename in sorted(os.listdir(source_dir)) if any(filename.lower().endswith(extension) for extension in supported_extensions) ]

        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), source_dir)

        total_volume = len(input_file_list)

        if offset<0:        # support offsets "from the right"
            offset += total_volume

        volume = int(volume_str) if len(volume_str)>0 else total_volume-offset
        if index_file:
            selected_filenames = input_file_list
        else:
            selected_filenames = input_file_list[offset:offset+first_n]


    output_signatures = preprocess_files(selected_filenames,
                                         source_dir,
                                         destination_dir,
                                         resolution,
                                         data_type,
                                         convert_to_bgr,
                                         normalize_symmetric,
                                         normalize_lower,
                                         normalize_upper,
                                         subtract_mean,
                                         given_channel_means,
                                         given_channel_stds,
                                         quantized,
                                         quant_scale, 
                                         quant_offset, 
                                         convert_to_unsigned,
                                         convert_dtype_before_resize,
                                         new_file_extension)

    fof_full_path = os.path.join(destination_dir, fof_name)
    with open(fof_full_path, 'w') as fof:
        for filename in output_signatures:
            fof.write(filename + '\n')
    return __record_entry__
