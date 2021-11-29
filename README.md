# Fake Job Postings Detection
## Directory structure 
```
├── README.md                         <- You are here
├── data                              <- Folder that contains data used 
├── notebooks/                        <- Folder that contains notebooks used in development
│   ├── eda.ipynb                     <- Notebook for EDA
│   ├── multimodal_modeling.ipynb     <- Notebook for testing multimodal modeling
├── src/                              <- Source scripts for data preprocessing and model training 
├── api.py                            <- Flask wrapper for running the model
├── client.py                         <- Client requesting inference
├── main.py                           <- Simplifies the execution of one or more of the src scripts
├── paper.pdf                         <- Project report
```

## Model Training

For model training via logistic regression, SVM, and CNN, open terminal and run command:
```bash
python main.py lr # logistic regression
python main.py svm # svm
python main.py cnn # cnn
python main.py multimodal --method svm # combinig text and meta features using svm
python main.py multimodal --method lr # combinig text and meta features using logistic regression
```
The best models for each architecture is saved into folder `output`. The best-performing model is `output/cnn_64_16_4.pickle`.

## API For Inference
For running the Flask App, please open two terminal windows, and run these commands separately:
```bash
python api.py
```
```bash
python client.py
```
By replacing text in json file in  "client.py", you can type in job descriptions and generate predicted probability.
For example, if you change `jd` to:
```python
jd = """The company is great, you will get a lot of money without working!!"""
```
It will return response like below:
```python
<Response [200]>
{'Job Description': 'The company is great, you will get a lot of money without working!!', 'Probability': '0.6580919781945342'}
```
