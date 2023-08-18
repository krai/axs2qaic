"""Control some hardware paramteters of Qaic cards via axs:

    - individual checking of vc (per device_id) :
            axs byname qaic_hardware_control , get get_vc --device_id=2

    - individual setting of vc (per device_id) :
            axs byname qaic_hardware_control , get set_vc --device_id=2 --vc_dec=12
            axs byname qaic_hardware_control , get set_vc --device_id=2 --vc_hex=0xc

    - setting the same vc value on multiple device_ids (returns a list of decimal previous vc values) :
            axs byname qaic_hardware_control , set_vcs --device_ids,=3,4 --vc_dec=12
"""

def set_vcs(device_ids, vc_hex, __entry__):
    "Due to current lack of iterators/mappers we have to iterate in Python"

    prev_vcs = []
    for device_id in device_ids:
        prev_vc = __entry__.call("get", ["get_vc"], { "device_id": device_id } )
        __entry__.call("get", ["set_vc"], { "device_id": device_id } )
        prev_vcs.append( prev_vc )

    return prev_vcs
