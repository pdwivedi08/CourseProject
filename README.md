# Tweet Classification Competition 

## About the Project

We've worked on the text classification competition project for the tweets. The train and test datasets are provided as part of competition and we intend to use current state-of-the-art machine learning NLP algorithms to beat the baseline on this text classiciation competition project.  

## Presentation Link on Youtube

Please refer below youtube link for the voice-over presentation for our project:

https://youtu.be/H1xQwJkV5cA


### Team members:

Harsangeet Kaur (kaur13@illinois.edu): Team Member
Pradeep Dwivedi (pkd3@illinois.edu): Team Lead

Our submission on the Leaderboard in Livelab can be found for the ID: pdwivedi08  


## Overview

This software can be used to classify tweets in Sarcasm and Not-Sarcasm categories. This can't be however, used for any other text classification or sentiment analysis with same level of accuracy or F1-score. 

This software achieves the high level of precision, recall and F1 score as against the generic transformers, since it has been especially trained on the tweet classification. 

## Implementation Documentation

We've made use of Google's T5 based fine-tuned transformer for twitter sarcasm detection. 

This model has been trained to identify sarcasm on tweets. We've used the Google Colab notebook to train the model and it took nearly 12 hours to train the model with the given data, on single GPU. 

We used the Trainer API from Huggingface to write the training code as it's easier to use. Also, we used the autotokenizer from the transformers library in Huggingfacce. 

We cleaned the test data for training and testing in such a way that the tweets are taken in correct sequence - first the orginal tweet and then it's responses in the chronolocial order. Also, we've removed all the filler words using regex and regular python functions from the tweets before using them for training and testing. 

We defined a function eval_conversation, to evaluate the curated tweets one-by-one and provide the output in Sarcasm and Not-Sarcams categories. 

We tried support vector machines (SVM) and T5 based transformer for this project. We got following values of precision, recall and F1 score with both these algorithm:


               precision              recall                   f1
            
    SVM        0.48314606741573035    0.14333333333333334      0.22107969151670953
    T5 based   0.7030114226375909     0.7522222222222222       0.726784755770263
    
    
The second approach i.e. the use of T5 based transfomer helped us to beat the baseline. Our final execution matrics can be viewed on the Leaderboard in Livelab, for the id: pdwivedi08
            
            

## Usage Documentation

All the code of the software is written in the jupyter notebooks, which can be opened from Anaconda IDE. 'classifyTweets.ipynb' is the main notebook which has the code to execute the test dataset. The 'test.jsonl' is stored inside the Data directory and the directory is included in the github. All other libraries needed to execute this code, are part of the 'classifyTweets.ipynb' notebook and would be imported when the notebook is executed. Therefore, no additional installation of any module is needed. 

The project github has the video demostration of the code execution as well and that can be used to install and run this software. Since the model is running T5 based transformer and the code has few displays, it will take around 5-7 minutes for the execution of the whole notebook on a macbook pro of 8 GB memory. The execution speed in-general will vary based on the hardware of the machine, used for running the notebook. 

For any further question related to the installation or the working of the software, please contact our team, at the below email IDs:

Harsangeet Kaur (kaur13@illinois.edu)
Pradeep Dwivedi (pkd3@illinois.edu)


## Detail of the Contributions of Team-members

Our team didn't has any prior background in natural languange processing(NLP) or machine learning (ML). Therefore, we started with understanding ML in the context of NLP and reading about it, online and on forums. Huggingface.co greatly helped us understanding the deep learning aspect of ML on NLP. 

Harsangeet Kaur tried support vector machine algorithm for text classification whereas Pradeep Dwivedi tried transformers for solving the problem. 

Together, we worked on the data cleaning, model training, software documentation and preparing the final presentation. 


## References

1. https://huggingface.co/mrm8488/t5-base-finetuned-sarcasm-twitter
2. stackoverflow.com
3. https://huggingface.co/transformers/main_classes/trainer.html
4. https://www.w3schools.com/python/python_regex.asp
5. https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34