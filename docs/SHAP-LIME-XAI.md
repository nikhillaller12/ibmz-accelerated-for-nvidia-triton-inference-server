# SHAP and LIME Explainability for Triton

This guide describes the SHAP and LIME explainability feature available with IBM Z Accelerated for NVIDIA Triton Inference Server. XAI is computed by the model backend and can be returned with the normal inference response. No separate XAI endpoint is required.

> **Important:** Support and configuration differ by backend. Follow the section for the backend serving your model. The Python backend is user-implemented: the runtime does not automatically calculate SHAP or LIME from configuration parameters.

## Overview

- Request the primary prediction and, when configured, SHAP and LIME values in the same KServe v2 inference request.
- Attribution values are exposed as output tensors. Default names are shap_values and lime_values.
- The documented attribution tensor layout is FP64 with shape [batch_size, n_features]. Declare output names, datatypes, and dimensions consistently in config.pbtxt and the inference request.
- XAI tensors are additional outputs; the primary prediction remains the model's regular output.

## Backend support

| Backend | Integration | Configuration notes |
| --- | --- | --- |
| Snap ML C++ (ibmsnapml) | Backend computes SHAP and LIME. | Reads XAI parameters from config.pbtxt. Output tensor declarations may be automatically appended when omitted; explicit declarations are recommended for clarity. |
| ONNX-MLIR (onnxmlir) | Backend computes SHAP and LIME. | Reads XAI parameters from config.pbtxt. Declare the SHAP and LIME output tensors in the model configuration. |
| PyTorch (pytorch) | PyTorch backend integration computes SHAP and LIME. | XAI flags are resolved when the model is loaded. Declare output tensors in the model configuration. |
| Python (python) | User implements SHAP and LIME in model.py. | The Python backend does not automatically parse or act on XAI parameters. The model code must produce outputs matching config.pbtxt. |

## Snap ML C++ backend (ibmsnapml)

The Snap ML C++ backend can calculate SHAP and LIME during inference. Enable the desired feature with the model parameters below and request the corresponding output tensor.

## ONNX-MLIR backend (onnxmlir)

The ONNX-MLIR backend can calculate SHAP and LIME during inference. Its XAI output tensors must be declared in config.pbtxt; ensure the names match those requested by the client.

## PyTorch backend (pytorch)

The PyTorch backend integrates XAI into its model execution flow. XAI flags are resolved during model initialization, and attribution values are calculated for inference requests when enabled. Declare the output tensors in config.pbtxt.

## Python backend (python)

For the Python backend, implement explanation generation in model.py and return the attribution tensors as part of the normal inference response. Libraries such as SHAP or LIME may be used by model code when available in the runtime environment. The backend does not automatically implement explainability or interpret the parameters below.

## Configuration

For the automatically integrated backends (ibmsnapml, onnxmlir, and pytorch), the supported parameters are:

| Parameter | Default | Purpose |
| --- | --- | --- |
| COMPUTE_SHAP | false | Enable SHAP attribution calculation. |
| COMPUTE_LIME | false | Enable LIME attribution calculation. |
| SHAP_OUTPUT_TENSOR_NAME | shap_values | Name of the SHAP output tensor; must match the output declaration and requested output. |
| LIME_OUTPUT_TENSOR_NAME | lime_values | Name of the LIME output tensor; must match the output declaration and requested output. |

Example parameter blocks (protobuf):

    parameters {
      key: "COMPUTE_SHAP"
      value { string_value: "true" }
    }
    parameters {
      key: "COMPUTE_LIME"
      value { string_value: "true" }
    }

Declare output tensors with names, datatypes, and dimensions that match the model and backend. The following illustrates the usual attribution tensor declarations; adjust dims to the number of input features and the model's batching configuration.

    output {
      name: "shap_values"
      data_type: TYPE_FP64
      dims: [ -1 ]
    }
    output {
      name: "lime_values"
      data_type: TYPE_FP64
      dims: [ -1 ]
    }

The dimensions above are illustrative. Validate the exact configuration against the model's expected feature count and Triton model configuration rules. For the Python backend, define and return these outputs in user code; the parameters above do not enable XAI automatically.

## Requesting explanations

Use Triton's normal KServe v2 inference endpoint and request the outputs by name. Example request body:

    {
      "inputs": [{
        "name": "IN0",
        "shape": [1, 5],
        "datatype": "FP32",
        "data": [[0.1, 0.2, 0.3, 0.4, 0.5]]
      }],
      "outputs": [
        {"name": "OUT0"},
        {"name": "shap_values"},
        {"name": "lime_values"}
      ]
    }

Replace IN0, its shape and data, OUT0, and the feature count with the names and shapes defined by your model. Only request XAI outputs that are configured and produced by the backend.

## Operational notes

- XAI outputs must be declared in the model configuration and requested by the client.
- Output tensor names must match between config.pbtxt and the request.
- The Python backend requires user-written explanation logic.
- In the C++ backends, XAI calculation errors are documented as non-fatal: the primary prediction is still returned and the XAI output can be zero-filled with a warning. Check server logs when attribution values are unexpected.
- Explanation values and their interpretation depend on the model, input representation, and backend implementation.
