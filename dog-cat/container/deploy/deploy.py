import argparse
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
import yaml
from determined.common.experimental import ModelVersion
from determined.experimental import Determined
from determined.pytorch import load_trial_from_checkpoint_path
from google.cloud import storage
from kserve import (
    KServeClient,
    V1beta1InferenceService,
    V1beta1InferenceServiceSpec,
    V1beta1PredictorSpec,
    V1beta1TorchServeSpec,
    constants,
    utils,
)
from kubernetes import client
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# =====================================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy a model to KServe")
    parser.add_argument("--deployment-name", type=str, help="Name of the resulting KServe InferenceService")
    parser.add_argument("--cloud-model-host", type=str, help="aws and gcp supported currently for storing model artifacts", choices=['gcp', 'aws'])
    parser.add_argument("--cloud-model-bucket", type=str, help="Cloud Bucket name to use for storing model artifacts")
    parser.add_argument("--google-application-credentials", type=str, help="Path to Google Application Credentials file", default=None)
    return parser.parse_args()


# =====================================================================================


def wait_for_deployment(KServe, k8s_namespace, deployment_name, model_name):
    while KServe.is_isvc_ready(deployment_name, namespace=k8s_namespace) == False:
        print(f"Inference Service '{deployment_name}' is NOT READY. Waiting...")
        time.sleep(5)
    print(f"Inference Service '{deployment_name}' in Namespace '{k8s_namespace}' is READY.")
    response = KServe.get(deployment_name, namespace=k8s_namespace)
    print(
        "Model "
        + model_name
        + " is "
        + str(response["status"]["modelStatus"]["states"]["targetModelState"])
        + " and available at "
        + str(response["status"]["address"]["url"])
        + " for predictions."
    )


# =====================================================================================


def get_version(client, model_name, model_version) -> ModelVersion:

    for version in client.get_model(model_name).get_versions():
        if version.name == model_version:
            return version

    raise AssertionError(f"Version '{model_version}' not found inside model '{model_name}'")


# =====================================================================================


def create_scriptmodule(det_master, det_user, det_pw, model_name, pach_id):

    print(f"Loading model version '{model_name}/{pach_id}' from master at '{det_master}...'")

    if os.environ["HOME"] == "/":
        os.environ["HOME"] = "/app"

    os.environ["SERVING_MODE"] = "true"

    start = time.time()
    client = Determined(master=det_master, user=det_user, password=det_pw)
    version = get_version(client, model_name, pach_id)
    checkpoint = version.checkpoint
    checkpoint_dir = checkpoint.download()
    trial = load_trial_from_checkpoint_path(checkpoint_dir, map_location=torch.device("cpu"))
    end = time.time()
    delta = end - start
    print(f"Checkpoint loaded in {delta} seconds.")

    print(f"Creating ScriptModule from Determined checkpoint...")

    # Create ScriptModule
    m = torch.jit.script(trial.model)

    # Save ScriptModule to file
    torch.jit.save(m, "scriptmodule.pt")
    print(f"ScriptModule created successfully.")


# =====================================================================================


def create_mar_file(model_name, model_version):
    print(f"Creating .mar file for model '{model_name}'...")
    os.system(
        "torch-model-archiver --model-name %s --version %s --serialized-file ./scriptmodule.pt --handler ./dog_cat_handler.py --force"
        % (model_name, model_version)
    )
    print(f"Created .mar file successfully.")


# =====================================================================================


def create_properties_file(model_name, model_version):
    config_properties = """inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071
enable_envvars_config=true
install_py_dep_per_model=true
enable_metrics_api=true
metrics_format=prometheus
NUM_WORKERS=1
number_of_netty_threads=4
job_queue_size=10
model_store=/mnt/models/model-store
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"%s":{"%s":{"defaultVersion":true,"marName":"%s.mar","minWorkers":1,"maxWorkers":5,"batchSize":1,"maxBatchDelay":5000,"responseTimeout":120}}}}""" % (
        model_name,
        model_version,
        model_name,
    )

    conf_prop = open("config.properties", "w")
    n = conf_prop.write(config_properties)
    conf_prop.close()

    model_files = ["config.properties", str(model_name) + ".mar"]

    return model_files


# =====================================================================================


def upload_model(model_name, files, cloud_provider, bucket_name):
    print(f"Uploading model files to model repository to cloud provider {cloud_provider} in bucket {bucket_name}...")
    if cloud_provider.lower() == 'gcp':
        upload_model_to_gcs(model_name, files, bucket_name)
    elif cloud_provider.lower() == 'aws':
        upload_model_to_s3(model_name, files, bucket_name)
    else:
        raise Exception(f"Invalid cloud provider {cloud_provider} specified")


def upload_model_to_s3(model_name, files, bucket_name):
    import boto3
    storage_client = boto3.client('s3')
    for file in files:
        if "config" in str(file):
            folder = "config"
        else:
            folder = "model-store"

        prefix = f'{model_name}/{folder}/'
        storage_client.upload_file("./" + file, bucket_name, prefix+file)

    print("Upload to S3 complete.")


def upload_model_to_gcs(model_name, files, bucket_name):
    storage_client = storage.Client()
    
    bucket = storage_client.get_bucket(bucket_name)

    for file in files:
        if "config" in str(file):
            folder = "config"
        else:
            folder = "model-store"
        blob = bucket.blob(model_name + "/" + folder + "/" + file)
        blob.upload_from_filename("./" + file)

    print("Upload to GCS complete.")


# =====================================================================================


def create_inference_service(kclient, k8s_namespace, model_name, deployment_name, pach_id, replace: bool):

    kserve_version = "v1beta1"
    api_version = constants.KSERVE_GROUP + "/" + kserve_version

    isvc = V1beta1InferenceService(
        api_version=api_version,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(
            name=deployment_name,
            namespace=k8s_namespace,
            annotations={"sidecar.istio.io/inject": "false", "pach_id": pach_id},
        ),
        spec=V1beta1InferenceServiceSpec(
            predictor=V1beta1PredictorSpec(
                pytorch=(V1beta1TorchServeSpec(storage_uri="gs://kserve-models/%s" % (model_name)))
            )
        ),
    )

    if replace:
        print(f"Replacing InferenceService with new version...")
        kclient.replace(deployment_name, isvc)
        print(f"InferenceService replaced with new version '{pach_id}'.")
    else:
        print(f"Creating KServe InferenceService for model '{model_name}'.")
        kclient.create(isvc)
        print(f"Inference Service '{deployment_name}' created.")


# =====================================================================================


def check_existence(kclient, deployment_name, k8s_namespace):

    print(f"Checking if previous version of InferenceService '{deployment_name}' exists...")

    try:
        response = kclient.get(deployment_name, namespace=k8s_namespace)
        exists = True
        print(f"Previous version of InferenceService '{deployment_name}' exists.")
    except (RuntimeError):
        exists = False
        print(f"Previous version of InferenceService '{deployment_name}' does not exist.")

    return exists


# =====================================================================================


class DeterminedInfo:
    def __init__(self):
        self.master = os.getenv("DET_MASTER")
        self.username = os.getenv("DET_USER")
        self.password = os.getenv("DET_PASSWORD")


# =====================================================================================


class KServeInfo:
    def __init__(self):
        self.namespace = os.getenv("KSERVE_NAMESPACE")


# =====================================================================================


class ModelInfo:
    def __init__(self, file):
        print(f"Reading model info file: {file}")
        info = {}
        with open(file, "r") as stream:
            try:
                info = yaml.safe_load(stream)

                self.name = info["name"]
                self.version = info["version"]
                self.pipeline = info["pipeline"]
                self.repository = info["repo"]

                print(
                    f"Loaded model info: name='{self.name}', version='{self.version}', pipeline='{self.pipeline}', repo='{self.repository}'"
                )
            except yaml.YAMLError as exc:
                print(exc)


# =====================================================================================


def main():
    args = parse_args()
    det = DeterminedInfo()
    ksrv = KServeInfo()
    model = ModelInfo("/pfs/data/model-info.yaml")

    if args.google_application_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.google_application_credentials

    print(f"Starting pipeline: deploy-name='{args.deployment_name}', model='{model.name}', version='{model.version}'")

    # Pull Determined.AI Checkpoint, load it, and create ScriptModule (TorchScript)
    create_scriptmodule(det.master, det.username, det.password, model.name, model.version)

    # Create .mar file from ScriptModule
    create_mar_file(model.name, model.version)

    # Create config.properties for .mar file, return files to upload to GCS bucket
    model_files = create_properties_file(model.name, model.version)

    # Upload model artifacts to Cloud  bucket in the format for TorchServe
    upload_model(model.name, model_files, args.cloud_model_host, args.cloud_model_bucket)

    # Instantiate KServe Client using kubeconfig
    k8s_config_file = "/determined_shared_fs/k8s.config"
    if os.path.exists(k8s_config_file):
        print ('k8s_config_file exists')
        kclient = KServeClient(config_file=k8s_config_file)
    else:
        kclient = KServeClient()

    # Check if a previous version of the InferenceService exists (return true/false)
    replace = check_existence(kclient, args.deployment_name, ksrv.namespace)

    # Create or replace inference service
    create_inference_service(kclient, ksrv.namespace, model.name, args.deployment_name, model.version, replace)

    # Wait for InferenceService to be ready for predictions
    wait_for_deployment(kclient, ksrv.namespace, args.deployment_name, model.name)

    print(f"Ending pipeline: deploy-name='{args.deployment_name}', model='{model.name}', version='{model.version}'")


# =====================================================================================


if __name__ == "__main__":
    main()
