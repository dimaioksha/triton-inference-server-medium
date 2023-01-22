from tritonclient.utils import np_to_triton_dtype
import tritonclient.http as httpclient
import sys
from PIL import Image
import numpy as np

model_name = "yolov5"
request_count = 2

with httpclient.InferenceServerClient(url="triton_server:8000", concurrency=request_count) as client:
    img = np.asarray(Image.open("./2_faces.jpeg"))
    # input0_data = np.array([np.asarray(Image.open("./2_faces.jpeg")), np.asarray(Image.open("./2_faces.jpeg"))])
    # input0_data = np.expand_dims(np.asarray(Image.open("./2_faces.jpeg")), 0)
    img_daun = img + np.random.normal(size=img.shape, loc=50, scale=1)
    print(np.expand_dims(img, 0).shape)
    Image.fromarray(img_daun.astype(np.uint8)).save("./2_faces_daun.jpeg")
    input0_data = np.array([img, img + np.random.normal(size=img.shape, loc=50, scale=1)]).astype(np.uint8)

    print(input0_data.shape)
    inputs = [
        httpclient.InferInput("INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)),
    ]

    inputs[0].set_data_from_numpy(input0_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    async_requests = []

    for i in range(request_count):
        # Asynchronous inference call.
        async_requests.append(
            client.async_infer(model_name=model_name,
                                inputs=inputs,
                                outputs=outputs))

    for async_request in async_requests:
        # Get the result from the initiated asynchronous inference request.
        # Note the call will block till the server responds.
        print(type(async_request))
        result = async_request.get_result()

        #print(result.get_response())
        # Validate the results by comparing with precomputed values.
        output0_data = result.as_numpy('OUTPUT0')

        print(f"OUTPUT0_DATA: {output0_data}, shape: {output0_data.shape}")

    #response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    #result = response.get_response()
    #output0_data = response.as_numpy("OUTPUT0")

    # print(f"OUTPUT0_DATA: {output0_data}, shape: {output0_data.shape}")
    #print(output0_data.shape)
    #print(output0_data[0, :, :].astype(np.uint8))
    #print(output0_data[1, :, :].astype(np.uint8))

    sys.exit(0)
