# Scalable-ML-Azure
 A real-time scalable machine learning service built with Azure

## Setup
Clone and add this repo into Azure DevOps. Detailed instructions on building this Azure Machine Learning service will be written in a blog article.


## Background & Data
This project is part of the our team's final project for Microsoft AI Challenge 2019. 
The data we used can be obtained from [here](https://www.kaggle.com/c/LANL-Earthquake-Prediction).


## Technology
- Azure Active Directory, Virtual Network, Secret scope, Key-Vault
- Azure DevOps, Azure Container Instance, Azure Kubeneters Service
- Azure Machine Learning Service, Databricks, Blob Storage
- MMLSPark


## Functions

- Prepare environment for model creation.
- Build model in Databricks.
- Create images from model.
- Deploy image to Azure Container Instance and Azure Kubernete Service.
- Test ACI endpoint

## Workflow

![](/docs/Azure_Overview_Full.png)

1. Load historical data from Blob storage to Databricks

2. Train LightGBM model use MML Spark on Databricks

3. Save and register the model into an AzureML Workspace Model Registry

4. Define model serving script using MML Spark Serving

5. Register model serving image in Azure Container Registry

6. Define AKS cluster instances and sizing to meet serving requirements

7. Deploy serving webservice to AKS

## Feature Engineering & Modeling

1. **Acoustic Signal Manipulation**

   Add a constant noise to each 150k segment (both in train and test) by calculating `np.random.normal(0, 0.5, 150000)`. Subtract the median of the segment after noise addition.

2. **Features**

    (i) number of peaks of at least support 2 on the denoised signal
    (ii) 20% percentile on std of rolling window of size 50
    (iii) 4th and (iv) 18th Mel-frequency cepstral coefficients mean. 


3. **Cross Validation**

   Shuffled 3-fold CV

4. **Model**

   LightGBM in MMLSpark

## DevOps

![](/docs/Azure_DevOps.png)

- Azure DevOps is used to continuously build the image of the model and release it as an endpoint
- Continuously deploy image as Azure Container Instance for test and Azure Kuberneters Service for production.
- Utilize Azure Active Directory, Virtual Network, Secret scope, Key-Vault to secure data and secret variables.
