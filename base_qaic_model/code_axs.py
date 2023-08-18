import sys
import json
import qaic_tool_parser
from qaic_tool_parser import Parse, Parse_Parametric, Parse_Csv

def parse_command(qaic_tool_chain_path, onnx_model_source, output_file_path,
                  node_precision_info, profile, input_list_file, model_name, sut_data, 
                  batchSize=None, percentile_calibration_value=None, cores=None, mos=None, ols=None, deallocDelay=None, splitSize=None, vtcmRatio=None,
                  __entry__ = None):
    """ We source most data from sut_data, and allowed a specific set of them to be overrided. 
    
    Please also beaware there are 2 device_id in sut_entry.
        axs byquery sut_config,sut=dyson , get device_id                     d0d3
        axs byquery sut_config,sut=dyson , get data , get device_id          0,3

Usage examples :

    """

    cmd_compile_list = [qaic_tool_chain_path + '/exec/qaic-exec']
    cmd_compile_list.append('-m={}'.format(onnx_model_source))
    cmd_compile_list.append('-aic-binary-dir={}'.format(output_file_path))

    assert sut_data , f"\n\n\nERROR: No sut data avalible!\n\n\n"
    for flags in ["cores", "mos", "ols", "deallocDelay", "splitSize", "vtcmRatio"]:
        val = eval(flags)
        if val == "None" or val == "null" or val == "":
            print(f"WARNING: Guessing user want to set {flags} as None, set it as None.")
            val = None
        sut_data.update({flags: eval(flags)})

    if percentile_calibration_value:
        sut_data.update({"percentile_calibration_value": percentile_calibration_value})

    compiler_flags, model_config = Parse_Csv("","").separate_param_into_csv_related(sut_data)

    for key,value in compiler_flags.items():
        compile_cmd = Parse_Csv(key,value)
        cmd_compile_list.append(compile_cmd.get())

    for key,value in model_config.items():
        if not value:
            compile_cmd = Parse(key)
        else:
            compile_cmd = Parse_Parametric(key,value)
        cmd_compile_list.append(compile_cmd.get())

    if node_precision_info: #in model_config.keys():
        cmd_compile_list.append('-node-precision-info={}'.format(node_precision_info))

    if profile:
        cmd_compile_list.append('-load-profile={}'.format(profile))

    #cmd_compile_list.append('-device-id={}'.format(sut_data["device_id"]))

    if (model_name == "bert-99.9"): #TODO: Move to config_for_model/ config_for_device?
        cmd_compile_list.append('-time-passes -aic-perf-warnings -aic-perf-metrics')


    cmd_compile = ' '.join(cmd_compile_list)
    return cmd_compile

def get_config_from_sut(config, default_val=None, sut_data=None, sut_entry=None):
    try:
        print(f"Setting {config} with {sut_entry.get_path()} ...")
        if config == "ml_model_seq_length":
            return sut_data["onnx_define_symbol"]["seg_length"]
        return sut_data[config]
    except:
        print(f"Bailing, set {config} to [{default_val}] ...")
        return default_val
