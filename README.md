# cat_indivisual_snn_module
cat_indivisual_snn_module

2021-12-03 
Modularization of predict part only
referenced code : https://www.kaggle.com/jovi1018/cat-individual-snn

development environment
python version : 3.6.8
needed library : tensorflow(version=2.6.2), numpy, matplotlib, sklearn, cv2, tqdm, os

Predict part of the SNN model trained on 10 cats is modularized.
When an image is input, the distance is calculated by comparing it with the 10 trained cats, and the cat with the smallest distance is predicted.
You can output test images and anchor images with plotImg, plotAnchor, and plotShow functions.
For detailed execution method, refer to module_test.py file.
