# Comparative Sentence Classification for Argument Mining

This repository contains the code related to the paper on [Tennis is Pretty Hard, but Still Easier than Basketball and Football. Categorization of Comparative Sentences in the Wild](https://arxiv.org/abs/1809.06152) at the ACL 2019 workshop on Argument Mining. This paper presents a set of machine learning classifiers which perform categorization of sentences into three classes: ``BETTER``, ``WORSE``, or ``NONE`` wehere each sentence is expected to contain mentions of two objects under a comparison. For instance, consider the following sentence: 
``Python is better than Ruby for scientific programming``. In this sentence, Python is compared to Ruby with respect to the aspect "scientific programming". Our classifier is expected to categorize it with the ``BETTER`` label as the first mentioned object is better than the second mentioned objext. Based on such comparisons one can compare objects in general. 

This repository contains also the result of crowdsourcing annotation campaign to create a training dataset, and the result of serveral classification experiments. In the experiments we rely on some libraries:

* [LexNet](https://github.com/vered1986/LexNET) [with some modifications](https://github.com/ablx/LexNET)
* [InferSent](https://github.com/facebookresearch/InferSent)

The annotated dataset is available to download at https://zenodo.org/record/3237552#.XPUr_6SxVqI