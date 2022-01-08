# Cesarean delivery prediction
A retrospective cohort of 1287 women delivering at one institution from 2011-2016 with 42 available demographic and clinical variables were examined using random forest recursive feature elimination (RF-RFE) and support vector machine recursive feature elimination (SVM-RFE) to select variables for analysis. 

Subsequently, logistic regression (LR) was compared to machine learning algorithms including decision tree (DT), random forest (RF), support vector machine with the linear kernel (SVM-Lin), and support vector machine with the radial basis function kernel (SVM-RBF) in their ability to predict cesarean as a binary outcome. 

70% of the cohort was used to train the algorithm and 30% was used for validation, for a total of 50 iterations. The area under the receiver operator curve (AUC) was used to determine the ability of an algorithm to correctly classify the outcome and p< 0.05 was considered significant.

SVM-RFE was more inclusive compared to RF-RFE with regards to feature selection and determined 22 significant variables compared to 10 (p < 0.001). 

Compared to traditional logistic regression, machine learning algorithms were noted to be superior (Figure 2) with average area under the curve (AUC) of 0.51 for LR, as opposed to 0.80 for RF, 0.77 for DT, 0.77 for SVM-Lin, and 0.77 for SVM-RBF (p < 0.001). 
