# DGERN
A robust drug representation learning model for eliminating cell specificity in gene expression profile and its application  

Learning good drug representations is
important for drug repositioning and the understanding of
mechanisms of drug action. Leveraging gene expression
profiles and eliminating cell specificity can facilitate drug
representation learning. In this paper, we propose a fourstage deep learning model that aims for drug representation
learning based on integrating gene expressing profiles and
drug descriptors, abbreviated as “DGERN”. The stacked
autoencoder is employed for data dimensionality reduction;
the iterative clustering modular is used to eliminate gene
expression differences between cell lines; the subclass pretraining modular and the label classifier is utilized integrate
the therapeutic use information of MeSH into drug
representations. The drug vectors represented by DGERN are
used in the subsequent calculation and prediction tasks of drug
development. In the task of predicting drug-disease associations,
DGERN achieved the best performance. DGERN’s AUC with
random forest reached 0.67, exceeding 0.6 of the second-placed
one, drug target data; in the prediction of drug-drug interactions,
DGERN’s AUC with random forest reached 0.73, which is second
in comparison with many data represented by SDNE; in the task
of predicting drug side effects, DGERN has also achieved good
performance using various machine learning methods.
