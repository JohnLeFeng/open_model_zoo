import sys

import openvino as ov

def main():
    core = ov.Core()

    ov_model = core.read_model("rvm_640x360_stateful.xml")
    original_results = ov_model.get_results()

    ov_model.remove_result(original_results[0])
    ov.save_model(ov_model, "rvm_640x360_stateful_mask_only.xml")

if __name__ == "__main__":
    sys.exit(main())
