# ''best_model_results'' Description

This directory contains the results of the best model found by the `q(i)_model_training.ipynb` script. The models are saved in the `Q(i)` directory.

The `Q(i)` directory contains the following directories like `Ranking_VGG19_comparisonweights_FC_20230502` and `Comparison_VGG19_imagenet_Convfusion_dropout_230421`.
The directory name determines the model basic information, including:
- `Ranking` or `Comparison`: the model type
- `VGG19` or `ResNet50`: the backbone network
- `imagenet` or `comparisonweights`: the backbone network weights
- `FC` or `Conv`: the subnetwork type
- `dropout`: additional information
- `230421` or `20230502`: the date of the model trained

