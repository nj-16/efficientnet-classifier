# Efficientnet-classifier
**Link to github repo used for implementation reference:** https://github.com/lukemelas/EfficientNet-PyTorch

**Link to research paper used for reference:** https://proceedings.mlr.press/v97/tan19a.html?ref=jina-ai-gmbh.ghost.io

**Baseline accuracy on testing data** - ~94%–95%

**Final improved version accuracy on testing data** - ~96%–97%

**Improvements added:** Most important and relevant changes made are fine-tuning the last few layers of EfficientNet instead of the entire model, which made it better adapt to the dataset, without overfitting. Also added stronger data augmentation for more generalization. ALSO used a learning rate scheduler and applied early stopping to save processing time. This led to a noticeable boost in accuracy over the baseline. 

**Dataset used:** Dataset used is CIFAR-10 dataset. Loaded using Pytorch's inbuilt "torchvision.datasets.CIFAR10", which automatically handles downloading and preprocessing
