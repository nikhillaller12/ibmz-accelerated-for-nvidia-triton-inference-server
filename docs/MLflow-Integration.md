# MLflow Integration with Triton Inference Server

This guide describes the MLflow integration for IBM Z Accelerated for NVIDIA Triton Inference Server. It covers model deployment operations, experiment tracking, and Triton inference-statistics monitoring.

## Architecture overview

The documented setup runs two processes in one MLflow-enabled Triton container:

- Triton Inference Server, including HTTP, gRPC, and metrics endpoints.
- An MLflow tracking server, providing the tracking UI/API, model registry, and artifact store.

A Python backend model named mlflow_model exposes two adapter versions: version 1 handles deployment operations, and version 2 handles monitoring operations. Both use the mlflow-triton plugin. The custom MLflow paths are registered in Triton's HTTP router; they are not separate services.

## Ports

| Host port | Container port | Service |
| ---: | ---: | --- |
| 5000 | 5000 | MLflow tracking server |
| 8010 | 8000 | Triton HTTP |
| 8011 | 8001 | Triton gRPC |
| 8012 | 8002 | Triton metrics |

## Deployment API: /v2/mlflow_deploy

Supported operations:

- create: deploy a model from an MLflow model URI.
- update: replace the artifact for an existing deployment and reload the model.
- delete: unload a model and remove its deployment files.
- list: list plugin-managed deployments.
- get: retrieve deployment metadata.
- predict: send input to a deployed model through Triton's normal inference API.

Example create request:

    curl -s -X POST http://localhost:8010/v2/mlflow_deploy       -H 'Content-Type: application/json'       -d '{
        "operation": "create",
        "model_name": "sklearn_cancer",
        "model_uri": "models:/sklearn_cancer@champion"
      }'

The model URI can identify a registered model version or an alias that resolves to a version. The Triton model repository must be configured and writable by the container process.

## Monitoring API: /v2/mlflow_monitor

Supported operations:

- monitor with watch: false: collect a single window of Triton statistics and log a monitoring run to MLflow.
- monitor with watch: true: start a background watcher that logs metrics at the requested interval.
- stop: stop one watcher or all watchers.
- list: list active watchers.

Each watcher is keyed by experiment name. Multiple watchers can run concurrently and share the Triton statistics endpoint.

## Experiment tracking

Training code can use the standard MLflow SDK and point its tracking URI to the MLflow server, for example http://localhost:5000. Standard tracking workflows can log parameters, metrics, tags, artifacts, and models; registered models can then be referenced by deployment operations.

Triton-side monitoring writes monitoring runs to MLflow as well. Use distinct experiment names for training and performance monitoring to keep runs organized. Experiment naming is the user's responsibility.

## Monitoring metrics and limitations

The current monitoring design uses Triton's model statistics endpoints and records these five metrics per monitoring window:

- Inference count
- Error count
- Mean inference latency
- Mean queue latency
- Mean compute-inference latency

The monitoring worker calculates per-window deltas from Triton statistics and logs the results to MLflow, with window parameters and tags.

**Not supported in the current design:** input drift monitoring and prediction drift monitoring. Triton's statistics endpoint exposes request counts and latency statistics, not the input-feature distributions or prediction distributions needed to calculate these drift measures.

## Runtime and configuration

The documented container design uses configuration for the Triton model repository, Triton URL, and MLflow tracking URI. Ensure the repository path is writable, the Triton statistics endpoint is reachable, and the MLflow tracking server is accessible to training clients and the monitoring worker.

Typical host URLs with the example port mappings:

- MLflow UI and tracking API: http://localhost:5000
- Triton HTTP and custom MLflow APIs: http://localhost:8010
- Triton metrics: http://localhost:8012/metrics

The standard Triton inference endpoint remains available for model inference. Deployment prediction operations call the target model's normal inference path rather than recursively calling the MLflow adapter.

## Error handling notes

- Empty or missing request bodies are rejected by the HTTP layer.
- Invalid operation payloads and deployment failures are returned as JSON status errors by the adapter.
- Deployment creation rolls back deployment files if model loading fails.
- A model that does not become READY within the configured timeout causes the deployment operation to fail.
- Monitoring failures to reach Triton statistics or MLflow are logged as warnings; the watcher continues to the next interval where possible.
