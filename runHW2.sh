#!/bin/bash

echo "linear Regression"
cd LinearRegression
python3 linear_regression.py
cd ..
echo "Ensemble Learning"
cd EnsembleLearning
python3 emsemble_learning.py
echo "Data now avaliable in Data folders"
read
