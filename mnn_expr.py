import MNN
import MNN.expr as expr
import numpy as np
import sys
import time

def v3_inference(model_path, num_inferences):

    print("-----MNN expr-----")


    # Measure loading time
    start_time = time.time()

    config = {
        'precision': 'low',
        'backend': 0, 
        'numThread': 4
    }

    rt = MNN.nn.create_runtime_manager((config,))
    rt.set_mode(9)
    rt.set_hint(0, 20)

    net = MNN.nn.load_module_from_file(model_path, ["image"], ["imageOUT"], runtime_manager=rt)

    loading_time = time.time() - start_time
    print(f"MNN expr : Model loading time: {loading_time:.4f} seconds")

    image = np.random.rand(1, 3, 512, 512).astype(np.float32)
    input_var = MNN.expr.placeholder([1, 3, 512, 512])
    input_var.write(image)
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)

    # Measure inference time
    total_inference_time = 0
    for _ in range(num_inferences):
        start_time = time.time()
        output_var = net.forward([input_var])
        inference_time = time.time() - start_time
        total_inference_time += inference_time

    average_inference_time = total_inference_time / num_inferences
    print(f"MNN expr : Average inference time over {num_inferences} runs: {average_inference_time:.4f} seconds")
    #
    # # Measure inference time
    # start_time = time.time()
    # output_var = net.forward([input_var])
    # inference_time = time.time() - start_time
    # print(f"MNN expr : Inference time: {inference_time:.4f} seconds")

    # print(output_var[0])

    print("---------------------")

    return (loading_time, average_inference_time)