Instructions:

To get the public datasets, go to the links below:
    MIMIC-III: https://physionet.org/content/mimic-eicu-fiddle-feature/1.0.0/
    Adult: https://github.com/AissatouPaye/Fairness-in-Classification-and-Representation-Learning
    COMPAS: https://www.kaggle.com/danofer/compass

Fill in the values in directories.py with where the data are/where the results will be stored

To run, execute run.py with arguments --dataset and --experiemnt 
    Dataset names are listed on line 102 of run.py
    Experiment names are listed on line 113 of run.py
    To change which approaches to test, change the contents of the list "approaches" in run.py on line 134 
    All approach names are listed between lines 104-111 of run.py

To plot the results, execute process_results.py
    Make sure the date on line 289 of process_results.py matches the date on line 122 of run.py
    To plot results for the ablation and hyperparameter sensitivity experiments, execute process_results2.py
    Make sure the date on line 113 of process_results2.py matches the date on line 122 of run.py
