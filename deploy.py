import boto3
import time
import os

# ==================================================================
# ================ 从环境变量中读取配置 ============================
# ==================================================================

# 这些值将由CodeBuild的环境变量设置提供
AWS_ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID")
AWS_REGION = os.environ.get("AWS_REGION")
ROLE_ARN = os.environ.get("ROLE_ARN")

# 检查环境变量是否都已设置
if not all([AWS_ACCOUNT_ID, AWS_REGION, ROLE_ARN]):
    raise ValueError("错误：一个或多个环境变量未设置 (AWS_ACCOUNT_ID, AWS_REGION, ROLE_ARN)")

# ECR镜像URI是根据环境变量自动生成的
IMAGE_URI = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/churn-predictor:latest"

# 为你的SageMaker资源命名
MODEL_NAME = "churn-predictor-model"
ENDPOINT_CONFIG_NAME = "churn-predictor-endpoint-config"
ENDPOINT_NAME = "churn-predictor-endpoint"

# ==================================================================
# ====================== 部署逻辑开始 ==============================
# ==================================================================

print("Initializing Boto3 SageMaker client...")
sagemaker_client = boto3.client("sagemaker", region_name=AWS_REGION)

# --- 1. 创建SageMaker模型 ---
print(f"Creating SageMaker Model: {MODEL_NAME}...")
try:
    sagemaker_client.delete_model(ModelName=MODEL_NAME)
    print(f"Deleted existing model: {MODEL_NAME}")
except sagemaker_client.exceptions.ClientError:
    pass

create_model_response = sagemaker_client.create_model(
    ModelName=MODEL_NAME,
    PrimaryContainer={"Image": IMAGE_URI},
    ExecutionRoleArn=ROLE_ARN,
)
print(f"Model '{create_model_response['ModelArn']}' created successfully.")

# --- 2. 创建SageMaker端点配置 ---
print(f"Creating SageMaker Endpoint Configuration: {ENDPOINT_CONFIG_NAME}...")
try:
    sagemaker_client.delete_endpoint_config(EndpointConfigName=ENDPOINT_CONFIG_NAME)
    print(f"Deleted existing endpoint configuration: {ENDPOINT_CONFIG_NAME}")
except sagemaker_client.exceptions.ClientError:
    pass

create_endpoint_config_response = sagemaker_client.create_endpoint_config(
    EndpointConfigName=ENDPOINT_CONFIG_NAME,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": MODEL_NAME,
            "InitialInstanceCount": 1,
            "InstanceType": "ml.t2.medium",
            "InitialVariantWeight": 1.0,
        }
    ],
)
print(f"Endpoint configuration '{create_endpoint_config_response['EndpointConfigArn']}' created successfully.")


# --- 3. 创建、更新或重建SageMaker端点 (更完善的逻辑) ---
print(f"Managing SageMaker Endpoint: {ENDPOINT_NAME}...")
try:
    response = sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    status = response['EndpointStatus']
    print(f"Endpoint found with status: {status}")

    if status == 'Failed':
        print("Endpoint is in 'Failed' state. Deleting and recreating...")
        sagemaker_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
        waiter = sagemaker_client.get_waiter('endpoint_deleted')
        waiter.wait(EndpointName=ENDPOINT_NAME)
        print("Endpoint deleted. Now creating a new one...")
        raise sagemaker_client.exceptions.ClientError({}, "CreateEndpoint")

    elif status == 'InService':
        print("Endpoint is 'InService'. Updating...")
        sagemaker_client.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=ENDPOINT_CONFIG_NAME,
        )
        print(f"Endpoint update initiated for '{ENDPOINT_NAME}'.")
    
    else:
        print(f"Endpoint is in '{status}' state. Please wait for it to be 'InService' or 'Failed' before running again.")

except sagemaker_client.exceptions.ClientError:
    print("Endpoint does not exist. Creating...")
    sagemaker_client.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
    )
    print(f"Endpoint creation initiated for '{ENDPOINT_NAME}'.")

print("\nDeployment script finished. Check the SageMaker console for status.")