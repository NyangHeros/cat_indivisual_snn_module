# cat_indivisual_snn_module
cat_indivisual_snn_module

Modularization of predict part only (2021-12-03)
  referenced code : https://www.kaggle.com/jovi1018/cat-individual-snn

development environment
  OS : Windows 10 Pro
  GPU : NVDIA GeForce RTX 2080 SUPER
  Cuda : V11.2.152
  python version : 3.6.8
  needed library : tensorflow(version=2.6.2), numpy, matplotlib, sklearn, cv2, tqdm, os

Explanation
  Predict part of the SNN model trained on 10 cats is modularized.
  When an image is input, the distance is calculated by comparing it with the 10 trained cats, and the cat with the smallest distance is predicted.
  You can output test images and anchor images with plotImg, plotAnchor, and plotShow functions.
  For detailed execution method, refer to module_test.py file.
