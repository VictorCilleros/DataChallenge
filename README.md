DATA CHALLENGE 2023 

** Context **
This challenge concerns the text analysis problem using Natural Language Processing (NLP) and machine learning techniques. The dataset used here is part of a recent questionnaire performed during the first shutdown in France to survey to survey older adults (over the age of 75) on their perception of the Coronavirus crisis. To give you an idea of the types of questions asked, the first block of questions concerns the perception of the dangerousness of the Coronavirus. It includes three questions:

Do you see CORONAVIRUS as: no danger or low danger, moderate danger, serious danger.
On a scale of 0 to 10, how intense have you been over the past few days?
Tell us what are your concerns?
The first two are closed questions and will not be concerned by this challenge. However, the third question invites the subjects to express themselves openly with their own sentences and the analysis of the latter is the subject of this challenge. More precisely, the objective here will be to categorize each answer in one or more (here they are limited to two) among four categories (classes). This problem is known in literature under the name: multi label classification.


** Dataset Description **
You are provided with a number of questionnaire answers that have been tagged by human reviewers.
The answer categories of the question "Percevez-le danger comment ?" are:

    - Craintes liées à la santé propre du répondant
    - Craintes liées aux inquiétudes pour les autres et les relations sociales avec les proches
    - Craintes plus générales ou formelles (économie, prolongement du confinement, etc.)
    - Les non réponses ou réponses non classées.

A more detailed description can be found in Category_description file.
You must create a model which predicts a probability of each category for each answer (lines in X_test.csv).

    - X_train.csv - the training set, contains answers with their binary labels.
    - X_test.csv - the test set, you have to predict the probabilities of belonging of the answers to the categories.
    - Category_description.rtf - description of answer categories.
    - sample_submission.csv - a sample submission file in the correct format.
    - unlabeled_data.csv - optional additional data without labels in the same format as the test file. 
      It is not asking to provide predictions to these data.