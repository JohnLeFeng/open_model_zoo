import sys

import openvino as ov

from openvino.runtime.passes import Manager, MakeStateful

def main():
    #  -shape src[1,3,720,1280],r1[1,16,144,256],r2[1,20,72,128],r3[1,40,36,64],r4[1,64,18,32]
    # HEIGHT = 720
    # WIDTH = 1280

    HEIGHT = 360
    WIDTH = 640

    ov_input = {
        'src': [1, 3, HEIGHT, WIDTH],
        'r1': [1, 16, HEIGHT // 5, WIDTH // 5],
        'r2': [1, 20, HEIGHT // 10, WIDTH // 10],
        'r3': [1, 40, HEIGHT // 20, WIDTH // 20],
        'r4': [1, 64, HEIGHT // 40, WIDTH // 40],
    }

    ov_model = ov.convert_model("robust_video_matting_mobilenetv3.onnx", input=ov_input)

    prep = ov.preprocess.PrePostProcessor(ov_model)
    prep.input(0).tensor().set_layout(ov.Layout("NCHW"))
    prep.input(0).preprocess().scale([255, 255, 255])
    ov_model = prep.build()

    ov.save_model(ov_model, "rvm_{}x{}.xml".format(WIDTH, HEIGHT))

    tensor_names = {
        "r1": "rr1",
        "r2": "rr2",
        "r3": "rr3",
        "r4": "rr4",
    }
    manager = Manager()
    manager.register_pass(MakeStateful(tensor_names))
    manager.run_passes(ov_model)
    ov.save_model(ov_model, "rvm_{}x{}_stateful.xml".format(WIDTH, HEIGHT))

if __name__ == "__main__":
    sys.exit(main())
