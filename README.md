# Text Classification

Welcome to the Text Classification Project Repository!

## Getting started

To get started with this project in local, follow these steps:

### 1. **Build Docker Image**

1. **Build Docker Image**
   ```
   docker build --platform=linux/amd64 -t mobilebert_app:local -f _scm_container/ai-pii-detector/Dockerfile .
   ```

### 2. Start Local Server using Docker

1. **Modify the volumes in your docker-compose.yaml file**
   ```
   services:
       app:
           platform: linux/amd64
           build:
              context: ./app
              dockerfile: Dockerfile
           image: mobilebert_app:local
           ports:
               - "8080:8080"
           volumes:
               - {{ your mobilebert model path}}:/opt/ml/model
   ```
2. **Run docker compose up, it will create the mobilebert model folder for you, ignore the chown error**
   ```
   docker compose up
   ```
3. **Move your pretrained model to the path that you defined earlier**
   ```
   COPY to {{ your mobilebert model path }}
   ```
4. **Stop and Run docker compose up again, it should start app at port 8080**
   ```
   docker compose up
   ```

## To Retrained Model with new dataset

1. **Update your dataset in `model/src/dataset/ml_datasets.csv`**
2. **Create virtual environment and install dependencies**
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r model/requirements.txt
   ```
   > **_NOTE:_** To deactivate your virtual environment, just: `deactivate`
3. **Run train script to train the model with latest dataset.**

   ```
   cd model
   python train.py
   ```

   > **_NOTE:_** This may takes more than 20 minutes to train

4. **(Optional) Test your newly trained model with `test.py`**.
   You may change the test email in the test.py script.

   ```
   cd model
   python test.py
   ```

5. **(Optional) Explain your output with your newly trained model with `explain_*.py`**.
   You may change the test email in each explain\_\*.py script.

   ```
   cd model
   python explain_lime.py
   python explain_shap.py
   ```

   You could find the .html file under src/explanation

6. **Replaced the `.tar.gz` file with newly trained model, and commit to Bitbucket to be deployed to s3 bucket.**
   ```
   cd model
   python zip.py
   ```
