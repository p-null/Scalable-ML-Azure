# Scalable-ML-Azure
 A complete scalable machine learning project built with Azure

## Features
- Utilized Azure Active Directory, Virtual Network, Secret scope, Key-Vault to secure data and secret variables.
- Continuously built projects with Azure DevOps, Deploy image as Azure Container Instance for test and Azure Kubeneters Service for production.

## Workflow

![](/docs/Azure_Overview_Full.png)

1. Load historical data from Blob storage to Databricks

2. Train LightGBM model use MML Spark on Databricks

3. Save and register the model into an AzureML Workspace Model Registry

4. Define model serving script using MML Spark Serving

5. Register model serving image in Azure Container Registry

6. Define AKS cluster instances and sizing to meet serving requirements

7. Deploy serving webservice to AKS