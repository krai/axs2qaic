import json

class Parse:
    def __init__(self, key, value=None):
        self.key = str(key)
        self.value = str(value) if isinstance(value, int) else value
        self.compile_cmd = ''

    def replace_underscore_to_hypen(self,input_string):
        return input_string.replace('_','-')

    def add_hypen(self,input_string):
            return ''.join(('-',input_string))

    def get(self):
        if 'image' in self.key:
            self.compile_cmd
        elif 'onnx_define_symbol' in self.key:
            self.compile_cmd
        else:
            self.compile_cmd = self.replace_underscore_to_hypen(self.key)
            self.compile_cmd = self.add_hypen(self.compile_cmd)
        return self.compile_cmd

class Parse_Parametric(Parse):
    def replace_underscore_to_hypen(self,input_string):
        return input_string.replace('_','-')

    def add_hypen(self,input_string):
        return ''.join(('-',input_string))

    def get(self):
        if 'image' in self.key:
            self.compile_cmd
        elif 'onnx_define_symbol' in self.key:
            for key,value in self.value.items():
                self.compile_cmd = ' '.join((self.compile_cmd,'-onnx-define-symbol={},{}'.format(key,value)))
        else:
            self.compile_cmd = self.replace_underscore_to_hypen(self.key)
            self.compile_cmd = self.add_hypen(self.compile_cmd)
            self.compile_cmd = ''.join((self.compile_cmd,'={}'.format(self.value)))
        return self.compile_cmd

class Parse_Csv(Parse):
    def __init__(self,key,value):
        super().__init__(key,value)
        self.csv_commands = {
                "cores":"-aic-num-cores=",
                "mos":"-mos=",
                "ols":"-ols=",
                "batchSize":"-batchsize=",
                "deallocDelay":"-allocator-dealloc-delay=",
                "splitSize":"-size-split-granularity=",
                "vtcmRatio":"-vtcm-working-set-limit-ratio=",
                "sdpClusterSizes":"-sdp-cluster-sizes=",
                "instances": ""
        }

    def get(self):
        if 'instances' in self.key:
            self.compile_cmd
        elif 'mos' in self.key and self.value != "compiler-default":
            self.compile_cmd = self.csv_commands[self.key] + self.value.replace(", ",",").strip("[]")
        elif 'vtcmRatio' in self.key:
            self.compile_cmd = self.csv_commands[self.key] + str(float(self.value)/100)
        elif 'sdpClusterSizes' in self.key and self.value:
            self.compile_cmd = self.csv_commands[self.key] + self.value.replace(", ",",").strip("[]")
        elif 'sdpClusterSizes' in self.key and not self.value:  # there are probably some hidden characters generated in .csv because without adding this condition it would output '-sdp-cluster-sizes='
            self.compile_cmd
        else:
            self.compile_cmd = self.csv_commands[self.key] + self.value
        return self.compile_cmd

    def separate_param_into_csv_related(self, params):
        """
        Wrapper function for translating the whole list of config into two dictionaries: compiler_flags and model_config
        TODO: Update Parse class such that it can handle one dictionary instead of requiring them to be pre-separated
        """
        if not isinstance(params, dict):
            params = json.loads(params)

        filterByKey = lambda keys: {x: params[x] for x in keys}
        _compiler_flags_keys = ( self.csv_commands.keys() & params.keys() )
        _compiler_flags = filterByKey(_compiler_flags_keys)

        _model_config_keys = ( self.csv_commands.keys() ^ params.keys() ) & params.keys()
        _model_config_keys = [key for key in _model_config_keys if 'recommended' not in key] # Remove Kilt related configs
        _model_config_keys = [key for key in _model_config_keys if 'profiling_thread' not in key] # Reomve profiling_thread
        _model_config = filterByKey(_model_config_keys)

        _compiler_flags = {k: v for k, v in _compiler_flags.items() if v is not None}
        return _compiler_flags, _model_config