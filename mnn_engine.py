import MNN
import numpy as np
import time

def v2_inference(modelPath, num_inferences):

    print("-----MNN engine-----")

    # Measure loading time
    start_time = time.time()

    net = MNN.Interpreter(modelPath)
    # net.setCacheFile(".cachefile")

    # set 7 for Session_Resize_Defer, Do input resize only when resizeSession
    # net.setSessionMode(7)

    # set 9 for Session_Backend_Auto, Let BackGround Tuning
    net.setSessionMode(9)
    # set 0 for tune_num
    net.setSessionHint(0, 20)

    config = {}
    config['backend'] = "CPU"
    config['precision'] = "low"
    session = net.createSession(config)

    loading_time = time.time() - start_time
    print(f"MNN engine : Model loading time: {loading_time:.4f} seconds")


    # print("Run on backendtype: %d \n" % net.getSessionInfo(session, 2))

    image = np.random.rand(1, 3, 512, 512).astype(np.float32)

    tmp_input = MNN.Tensor((1, 3, 512, 512), MNN.Halide_Type_Float, \
                           image, MNN.Tensor_DimensionType_Caffe)

    # input
    inputTensor = net.getSessionInput(session)
    net.resizeTensor(inputTensor, (1, 3, 512, 512))
    net.resizeSession(session)
    inputTensor.copyFrom(tmp_input)

    # Measure inference time
    total_inference_time = 0
    for _ in range(num_inferences):

        start_time = time.time()

        # infer
        net.runSession(session)

        inference_time = time.time() - start_time
        total_inference_time += inference_time

    average_inference_time = total_inference_time / num_inferences
    print(f"MNN expr : Average inference time over {num_inferences} runs: {average_inference_time:.4f} seconds")
    #
    # # Measure inference time
    # start_time = time.time()
    #
    #
    # inference_time = time.time() - start_time
    # print(f"MNN engine : Inference time: {inference_time:.4f} seconds")

    outputTensor = net.getSessionOutput(session)

    # output
    # outputShape = outputTensor.getShape()
    # outputHost = createTensor(outputTensor)
    # outputTensor.copyToHostTensor(outputHost)

    # print("output : {}".format(outputTensor))

    print("---------------------")

    return (loading_time, average_inference_time)

