3.4 Gen AI Demo #4 - Machine Learning 
NOTE: Partners with an active Machine Learning Specialization may skip Demo #4. Machine Learning Specialization must not expire within six months from the date of Generative AI - Services Specialization application. If Machine Learning Specialization expires within 6 months, partner must complete Demo #4 and submit Customer Success Story aligned to technical solution described in Demo #4.
Example of an end-to-end Vertex AI Pipeline, where you use any of the Vertex AI supported frameworks,  using the Chicago taxi trips dataset (BigQuery) to improve its service. This demo requires the use of either Vertex AI or Kubeflow (deployed on Google Kubernetes Engine (GKE)), with data pre-processing performed using Dataflow, BigQuery, or Dataproc.  
The partner’s Machine Learning - Services Specialization capability will be assessed via one demo, (or active Machine Learning Specialization) which requires the following four artifacts: 1) Code and repository, 2) Data in Google Cloud, 3) Whitepaper, and 4) Proof of deployed model. 
Each demo needs to be presented as a customer-facing deliverable (meaning, these demos, when presented, will enable the customer to fully comprehend the partner’s Expertise in the Machine Learning - Services Specialization). 
This demo must align to an approved machine learning Customer Success Story with either Vertex AI or Kubeflow with data pre-processing performed using Dataflow, BigQuery, or Dataproc. The approved machine learning Customer Success Story may be used to achieve the required 10 Customer Success Story points for this Generative AI Services - Specialization application. Refer to Google Cloud Machine Learning best practices
Item
Requirement
Description
3.4.0 Demo #4
ML Customer Success Story Alignment CS-XXXXX
3.4.0 ML Evidence  - Insert CS-XXXXX and document links 
Align an approved Customer Success Story utilizing any of the Vertex AI supported frameworks; and also requires the use of either Vertex AI or Kubeflow (deployed on GKE), with data pre-processing performed using Dataflow, BigQuery, or Dataproc.
ML.3.4.1 Code
Partners must demonstrate the detailed capabilities outlined in the demo. The default is the partner makes a demo based on the scenario and data listed in the demo section. However, If partners can do this with a real-life customer engagement, and are able to share the code / documentation that demonstrates the same capabilities, that can be used to fulfill this requirement. If other datasets are utilized, please ensure the partner has acquired all of the appropriate licenses.
ML3.4.1.1 Code repository


Partners must provide a link to the code repository – for example, GitHub, GitLab, Google Cloud Certificate Signing Request (CSR), which includes a ReadMe file.
Evidence must include an active link to the code repository containing all code that is used in Demo #4. This code must be reviewable/readable by the assessor, and modifiable by the customer. In addition, the repository should contain a ReadMe file with code descriptions and detailed instructions for running the model/application.
ML 3.4.1.2 Code origin certification
Partners must certify to either of these two scenarios: 1) all code is original and developed within the partner organization, or 2) licensed code is used, post-modification.
Evidence must include a certification by the partner organization for either of the above code origin scenarios. In addition, if licensed code is used post-modification, the partner must certify that the code has been modified per license specifications.
ML 3.4.2 Data
ML 3.4.2.1 Dataset in Google Cloud
Partners must provide documentation of where the data of Demo #4 is stored within Google Cloud (for access by the models during training, testing, and in production).
Evidence must include the project name and project ID for the Google Cloud Storage bucket or BigQuery dataset with the data (for Demo #4).
ML 3.4.3 Whitepaper/ blog - describes the key steps of machine learning model development

Provide a link to a whitepaper or blog post (can be a duplicate of what is used for the capability assessment) describing how you ensure machine learning  projects address the security and privacy concerns associated with your machine learning efforts. For example, how do you ensure sensitive training data stored in Google Cloud is properly secured, do you consider de-identification (masking, bucketing, etc.) of datasets, and more?
ML 3.4.3.1 Business goal and machine learning solution


Partners must describe:
The business question/goal being addressed
The machine learning use case
How machine learning solution is expected to address the business question/goal
Evidence must include (in the whitepaper) a top-line description of the business question/goal being addressed in this demo, and how the proposed machine learning solution will address this business goal.
ML 3.4.3.2 Data exploration 


Partners must describe the following:
How and what type of data exploration was performed
What decisions were influenced by data exploration
Evidence must include a description (in the whitepaper) of the tools used and the type(s) of data exploration performed, along with code snippets (that achieve the data exploration). Additionally, the whitepaper must describe how the data/model algorithm/architecture decisions were influenced by the data exploration.
ML 3.4.3.3 Feature engineering


Partners must describe the following:
What feature engineering was performed
What features were selected for use in the machine learning model and why
Evidence must include a description (in the whitepaper) of the feature engineering performed (and rationale for the same), what original and engineered features were selected for incorporation as independent predictors in the machine learning model, and why. Evidence must include code snippets detailing the feature engineering and feature selection steps.
ML 3.4.3.4 Preprocessing and the data pipeline


The partner must describe the data preprocessing pipeline, and how this is accomplished via a package/function that is a callable API (that is ultimately accessed by the served, production model).
Evidence must include a description (in the whitepaper) of how data preprocessing is accomplished using Dataflow, BigQuery, and/or Dataproc, along with the code snippet that performs data preprocessing as a callable API.
ML 3.4.3.5 Machine learning model design(s) and selection


Partners must describe the following:
Which machine learning model/algorithm(s) were chosen for Demo #4
What criteria were used for machine learning model selection
Evidence must describe (in the whitepaper) selection criteria implemented, and the specific machine learning model algorithms that were selected for training and evaluation purposes. Code snippets detailing the model design and selection steps must be enumerated.
ML 3.4.3.6 Machine learning model training and development

TIP: Assessor will be looking for how the partner implements Google Cloud Machine Learning best practices.


Partners must document the use of Vertex AI or Kubeflow for machine learning model training, and describe the following:
Dataset sampling used for model training (and for independent dev/test datasets) and justification of sampling methods
Implementation of model training, including adherence to Google Cloud best practices for distribution, device usage, and monitoring
The model evaluation metric that is implemented, and a discussion of why the implemented metric is optimal given the business question/goal being addressed
Hyperparameter tuning and model performance optimization
How bias/variance were determined (from the train-dev datasets) and tradeoffs used to influence and optimize machine learning model architecture
Evidence must describe (in the whitepaper) each of the machine learning model training and development points (above). In addition, code snippets that perform each of these tasks need to be enumerated.
ML 3.4.3.7 Machine learning model evaluation


Partners must describe how the machine learning model, post-training, and architectural/hyperparameter optimization performs on an independent test dataset.
Evidence must include records/data (in the whitepaper) of how the machine learning model developed and selected to address the business question performance on an independent test dataset (that reflects the distribution of data that the machine learning model is expected to encounter in a production environment). In addition, code snippets on model testing need to be enumerated.
ML 3.4.4 Proof of deployment
ML 3.4.4.1 Model/ application on Google Cloud


Partners must provide proof that the machine learning model/application is deployed and served on Google Cloud with Vertex AI or Kubeflow.
Evidence must include the Project Name and Project ID of the deployed cloud machine learning model and client.
ML 3.4.4.2 Callable library/ application
Partners must demonstrate that the machine learning model for Demo #4 is a callable library and/or application.
Evidence must include a demonstration of how the served model can be used to make a prediction via an API call.
ML 3.4.4.3 Editable model/ application
Partners must demonstrate that the deployed model is customizable.
Evidence must include a demonstration that the deployed model is fully functional after an appropriate code modification, as might be performed by a customer.  

