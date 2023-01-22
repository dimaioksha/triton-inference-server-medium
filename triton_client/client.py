from tritonclient.utils import np_to_triton_dtype
import tritonclient.http as httpclient
import tritonclient
import numpy as np
import logging
import time
from typing import NoReturn

TRITON_STRING_TO_NUMPY = {
    "TYPE_BOOL": bool,
    "TYPE_UINT8": np.uint8,
    "TYPE_UINT16": np.uint16,
    "TYPE_UINT32": np.uint32,
    "TYPE_UINT64": np.uint64,
    "TYPE_INT8": np.int8,
    "TYPE_INT16": np.int16,
    "TYPE_INT32": np.int32,
    "TYPE_INT64": np.int64,
    "TYPE_FP16": np.float16,
    "TYPE_FP32": np.float32,
    "TYPE_FP64": np.float64,
    "TYPE_STRING": np.object_,
}

logger = logging.Logger(__name__)


class TritonClient:
    def __init__(self, model_name: str, url: str, verbose: bool = False, concurrency: int = 1):
        """
        model_name: Name of the model to interact with
        url: Inference server url to interact with using HTTP e.g. localhost:8000
        verbose: If True generate verbose output. Default value is False.
        concurrency: Number of parallel tasks which can be sent by this client
        """

        self._client = httpclient.InferenceServerClient(url=url, verbose=verbose, concurrency=concurrency)

        while not self._client.is_server_live():
            logger.warning(f"Inference server {url} is not alive")
            time.sleep(5)

        while not self._client.is_model_ready(model_name):
            logger.warning(f"Specified model {model_name} is not ready yet")
            time.sleep(5)

        self.model_name = model_name
        self.model_config = self._client.get_model_config(self.model_name)

    def _preprocess_batch(self, batch: np.ndarray) -> np.ndarray:
        if batch.ndim in [1, 2]:
            logger.error(f"Specified batch has few dimensions: need 3 or 4, you passed: {batch.ndim}")
        if batch.ndim == 3:
            batch = np.expand_dims(batch, 0)
            logger.info(f"Specified batch has 3 dimensions (one picture). Batch size is increased to 1")
        input_config = self.model_config["input"]
        if len(input_config) != 1:
            logger.error(f"{len(input_config)} size input is not supported")
        cast_type = input_config[0]["data_type"]
        batch = batch.astype(TRITON_STRING_TO_NUMPY[cast_type])

        return batch

    def send_task(self, batch: np.ndarray, model_version: str = "1") -> tritonclient.http.InferAsyncRequest:
        """
        Send async task to the server to preprocess batch
        Despite it operates with one element in inputs/outputs massive,
        It works with batch of information
        """

        batch = self._preprocess_batch(batch)
        inputs = [
            httpclient.InferInput(self.model_config["input"][0]["name"], batch.shape, np_to_triton_dtype(batch.dtype)),
        ]
        inputs[0].set_data_from_numpy(batch)
        outputs = [
            httpclient.InferRequestedOutput(self.model_config["output"][0]["name"]),
        ]

        async_request = self._client.async_infer(model_name=self.model_name,
                                                 model_version=model_version,
                                                 inputs=inputs,
                                                 outputs=outputs)

        return async_request

    def get_task_result(self, request: tritonclient.http.InferAsyncRequest) -> np.ndarray:
        """
        The call will block thread till the inference server responds
        """
        inference_result = request.get_result()
        output_data = inference_result.as_numpy(self.model_config["output"][0]["name"])
        return output_data

    def close(self) -> NoReturn:
        """
        Close the client. Any future calls to server will result in an Error
        """
        self._client.close()


if __name__ == "__main__":
    from PIL import Image
    client = TritonClient(model_name="yolov5",
                          url="triton_server:8000",
                          concurrency=10)
    np.random.seed(10)
    # You have to change this path to image you would like to test inference with
    path_to_example_image = "./2_faces.jpeg"
    img = np.asarray(Image.open(path_to_example_image))
    img_preprocessed = img + np.random.normal(size=img.shape, loc=50, scale=1)

    two_images_same = np.array([img, img])
    two_images_preprocessed = np.array([img, img_preprocessed])

    request = client.send_task(img)
    result = client.get_task_result(request)
    print(f"OUTPUT0_DATA: {result}, shape: {result.shape}")

    request = client.send_task(two_images_same)
    result = client.get_task_result(request)
    print(f"OUTPUT0_DATA: {result}, shape: {result.shape}")

    request = client.send_task(two_images_preprocessed)
    result = client.get_task_result(request)
    print(f"OUTPUT0_DATA: {result}, shape: {result.shape}")


