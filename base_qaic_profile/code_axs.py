import json
import os
import shutil
import qaic_tool_parser
from qaic_tool_parser import Parse, Parse_Parametric, Parse_Csv

# Calibration dataset.
def form_batches_for_profile(batchSize, output_file_path=None,
                             input_file_list=None, dataset_preprocessed_dir=None, dataset_preprocessed_fof=None, 
                             input_batches_fof=None, output_batches_dir=None):
    """ Form batches for profile
    
    Run with either first_n or index_file option but not both. TO run with the index_file option we need to full imagenet dataset (5000 image).

Usage examples : 
    axs byquery profile,model_name=resnet50,device=qaic,batchSize=8,first_n=100
    axs byquery profile,model_name=resnet50,device=qaic,batchSize=8,index_file=cal_image_list_option_1.txt,

    """

    batchSize = int(batchSize)

    images=os.path.join(dataset_preprocessed_dir,dataset_preprocessed_fof)
    print("Calibration images: ", images)

    output_dir = os.path.dirname(output_file_path)
    output_batches_dir = os.path.join(output_dir, output_batches_dir)
    os.mkdir(output_batches_dir)

    output_file_path_fof = os.path.join(output_batches_dir, input_batches_fof )

    i = 0
    lastset_list = []

    with open(output_file_path_fof, "a") as f:
        for filename in input_file_list:
            filename = filename.split(".")[0] + ".rgbf32"
            if i % batchSize == 0:
                newfilename = filename + ".raw"
                output_file_path = os.path.join(output_batches_dir,newfilename)

                f.write(output_file_path)
                f.write("\n")
            if i < batchSize:
                lastset_list.append(filename)

            dataset_preprocessed_path = os.path.join(dataset_preprocessed_dir,filename)
            assert os.path.isfile(dataset_preprocessed_path), f"\n\n\nERROR: No image file at {dataset_preprocessed_path}, please either download the full ImageNet Dataset or use the first_n option to compile the profile. \n\n\n"
            with open(output_file_path, "ab") as file_calibration, open(dataset_preprocessed_path, "rb") as file_input:
                print("Input: ", dataset_preprocessed_path, "; Output: ", output_file_path)
                file_calibration.write(file_input.read())
            i = i + 1
    len_lastset = len(lastset_list)

    j = 0

    while i%batchSize != 0:
        filename=lastset_list[ len_lastset - 1 - j ]
        dataset_preprocessed_path = os.path.join(dataset_preprocessed_dir,filename)

        with open( output_file_path, "ab") as file_calibration, open(dataset_preprocessed_path, "rb") as file_input:
            print("Input: ", dataset_preprocessed_path, "; Output: ", output_file_path)
            file_calibration.write(file_input.read())
        i = i + 1
        j = j + 1
    return output_file_path_fof

def parse_command(qaic_tool_chain_path, onnx_model_source, output_file_path, 
                  model_name, sut_data, batchSize, profiling_thread,
                  input_list_file=None, node_precision_info=None,
                  __record_entry__=None, __entry__=None):
    print("Starting to Compile Profile ...")

    cmd_profile_list = [qaic_tool_chain_path + '/exec/qaic-exec']

    cmd_profile_list.append('-m={}'.format(onnx_model_source))
    cmd_profile_list.append('-dump-profile={}'.format(output_file_path)) # rmb to teleport to the location before dumping


    cmd_profile_list.append('-profiling-threads={}'.format(profiling_thread))

    assert sut_data , f"\n\n\nERROR: No sut data avalible!\n\n\n"
    _, sut_data = Parse_Csv("","").separate_param_into_csv_related(sut_data)

    # Set input-list-file
    #if (model_name == "bert-99" or model_name == "resnet50"):
    if input_list_file is not None:
        assert os.path.exists(input_list_file) , f"\n\n\nERROR: Compiling profile for {model_name}, but missing input_list_file.\n\n\n"
        cmd_profile_list.append(f'-input-list-file={input_list_file}')

    # Set node_precision_info
    if node_precision_info:
        assert os.path.exists(node_precision_info), f"\n\n\nERROR: Compiling profile for {model_name}, but missing node_precision_info.\n\n\n"
        cmd_profile_list.append('-node-precision-info={}'.format(node_precision_info))

    if model_name == "resnet50":
        cmd_profile_list.append('-batchsize={}'.format(batchSize))

    # Update command with the values from sut_data.
    for key, value in sut_data.items():

        if not value:
            compile_cmd = Parse(key)
        else:
            compile_cmd = Parse_Parametric(key,value)
        cmd_profile_list.append(compile_cmd.get())

    cmd_profile = ' '.join(cmd_profile_list)

    return cmd_profile

def get_config_from_sut(config, default_val=None, sut_data=None, sut_entry=None):
    try:
        print(f"Setting {config} with {sut_entry.get_path()} ...")
        return sut_data[config]
    except:
        print(f"Bailing, set {config} to [{default_val}] ...")
        return default_val
