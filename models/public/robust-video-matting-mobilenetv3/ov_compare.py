import os
import sys

import numpy as np
import openvino as ov


def evaluate(reference_data, actual_data):
    actual_abs = np.abs(actual_data)
    reference_abs = np.abs(reference_data)
    diff_abs = np.abs((actual_data - reference_data))

    max_ref = np.max(reference_data)
    min_ref = np.min(reference_data)
    max_abs = np.max(reference_abs)

    max_diff = np.max(diff_abs)
    max_diff_index = np.unravel_index(diff_abs.argmax(), diff_abs.shape)

    ratio = diff_abs / max_abs
    diff_count = np.sum((ratio > 0.01))

    sum_ref = np.sum(reference_abs)
    sum_diff = np.sum(diff_abs)

    print ("Total Diff ratio = ", diff_count * 100 / np.prod(actual_abs.shape), "%")
    print ("Total Sum ratio = ", sum_diff * 100 / sum_ref, "%")
    print ("Single Max Diff ratio = ", max_diff * 100 / max_abs)
    print ("Max abs = ", max_abs)
    print ("Max diff abs = ", max_diff, ", ref = ", reference_data[max_diff_index], ", actual = ", actual_data[max_diff_index], ", index = ", max_diff_index)
    print ("ref min = ", min_ref, ", ref max = ", max_ref)


def main():
    iterations = 5

    core = ov.Core()

    original_model = core.compile_model("rvm_640x360.xml", "CPU")
    stateful_model = core.compile_model("rvm_640x360_stateful.xml", "CPU")

    original_input_info = original_model.inputs
    original_output_info = original_model.outputs

    original_input_names = []
    original_input_shapes = []
    for i in range(len(original_input_info)):
        original_input_names.append(original_model.input(i).get_any_name())
        original_input_shapes.append(original_model.input(i).get_shape())
    
    print (original_input_names)
    print (original_input_shapes)

    original_output_names = []
    for i in range(len(original_output_info)):
        original_output_names.append(original_model.output(i).get_any_name())

    print (original_output_names)

    original_input_tensors = {}
    stateful_input_tensors = {}

    original_infer_queue = original_model.create_infer_request()
    stateful_infer_queue = stateful_model.create_infer_request()
    for state in stateful_infer_queue.query_state():
        state.reset()

    for i in range(iterations):
        for _idx, _name in enumerate(original_input_names):
            if _name == 'src':
                original_input_tensors[_name] = np.random.random_sample(original_input_shapes[_idx]).astype(np.float32)
                stateful_input_tensors[_name] = original_input_tensors[_name]
            else:
                if i == 0:
                    original_input_tensors[_name] = np.zeros(original_input_shapes[_idx]).astype(np.float32)
                else:
                    original_input_tensors[_name] = original_output_tensors['r{}'.format(_name)]
        
        original_output_tensors = original_infer_queue.infer(original_input_tensors)
        stateful_output_tensors = stateful_infer_queue.infer(stateful_input_tensors)

    for _output_name in original_output_names:
        print ("\n" + _output_name + "\n")
        if _output_name == 'fgr' or _output_name == 'pha':
            evaluate(original_output_tensors[_output_name], stateful_output_tensors[_output_name])
        elif _output_name == 'rr1':
            evaluate(original_output_tensors[_output_name], stateful_infer_queue.query_state()[3].state.data)
        elif _output_name == 'rr2':
            evaluate(original_output_tensors[_output_name], stateful_infer_queue.query_state()[0].state.data)
        elif _output_name == 'rr3':
            evaluate(original_output_tensors[_output_name], stateful_infer_queue.query_state()[2].state.data)
        else:
            evaluate(original_output_tensors[_output_name], stateful_infer_queue.query_state()[1].state.data)

if __name__ == "__main__":
    sys.exit(main())