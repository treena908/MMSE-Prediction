# MMSE-Prediction

An automated system to predict MMSE (Mini Mental State Exam) scores reflecting individuals' cognitive health status, based on their free speech samples from conversational interviews. This project uses data from DementiaBank's Pitt corpus and its subset dataset from the ADDReSS challenge (http://www.homepages.ed.ac.uk/sluzfil/ADReSS/)and the raw data 
is not publicly available.

Data:

Inside the data folder, we have saved preprocessed data prepaperd from the actual conversational transcripts and audio file.

Model:

-Using the preprocessed data of ADDReSS dataset, we ran some traditional machine learning model to predict MMSE score on the given test set (SVR_model.py). (For more information, read out paper published in INTERSPEECH 2020).
- We also experimented with longituidinal prediction model of MMSE score using deep neural based architecture. (model_longituidinal.py)

Paper Citation:
Shahla Farzana* and Natalie Parde. Exploring MMSE Score Prediction Using Verbal and Non-Verbal Cues. To appear in the Proceedings of the 21st Conference of the International Speech Communication Association (INTERSPEECH 2020). Shanghai, China, October 25-29, 2020
