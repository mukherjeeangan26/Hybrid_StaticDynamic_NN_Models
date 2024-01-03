# Hybrid Static-Dynamic NN Models Library in MATLAB

MATLAB Codes for Training Different Candidate Hybrid All-Nonlinear Static-Dynamic Neural Network Models

Author: Angan Mukherjee (am0339@mix.wvu.edu)

Last Page Update: January 03, 2024

## Announcement

We are very welcome to your contribution. Please feel free to reach out to us with your feedback and suggestions 
on how to improve the current models.

## Publication

This repository contains MATLAB codes for training and validating different candidate neural network (NN) architectures
while modeling various nonlinear dynamic chemical process systems. The corresponding publication for this work is:

Mukherjee, A. & Bhattacharyya, D. Hybrid Series/Parallel All-Nonlinear Dynamic-Static Neural Networks: Development, Training, 
and Application to Chemical Processes. Ind. Eng. Chem. Res. 62, 3221–3237 (2023). 

Available online at: https://pubs.acs.org/doi/full/10.1021/acs.iecr.2c03339

These codes will be updated in subsequent versions to enhance robustness of the network architectures and user friendliness.

## Brief Description

### Sample Data

Two sample dynamic datasets have been provided in the 'data' folder. Note that the codes uploaded in this repository are generic
and can be applied to model any dynamic time-series data. The training and validation datasets need to be loaded at the beginning
while running the training / validation code, i.e., 'Run_Hybrid_NN_MainFile.m'. The rows of the input and output data matrices 
refer to the time indices (steps) while the columns signify the different input / output variables.

### Candidate Network Architectures

This code requires the MATLAB Neural Network and Deep Learning Toolbox Packages.

This code develops and compares the different NN architectures considered for modeling any generic nonlinear dynamic data.

The candidate network models considered in this code for comparison are:

  1. **Nonlinear Static (NLS) Network (Feedforward three-layered NN)**
  
  2. **Nonlinear Dynamic (NLD) Network (NARX-type Recurrent NN)**
  
  3. **Hybrid Series (NLS - NLD) Static-Dynamic NN**
  
  4. **Hybrid Series (NLD - NLS) Dynamic-Static NN**
  
  5. **Hybrid Parallel (NLS || NLD) Static-Dynamic NN**

At the beginning of running the MainFile, the user specifies the type of candidate model architecture to be trained from a list provided
in the Command Window.

Note that for the network structures involving a dynamic (NLD) model, the initial condition has to be specified during validation of the 
corresponding dynamic / hybrid static-dynamic models.

### NOTE

  * It is to be noted that the hybrid series and parallel all-nonlinear static-dynamic networks have shown satiafactory performances while
    modeling various nonlinear complex dynamic process systems. However, the specific types of series or parallel architecture which will
    qualify as the optimal model depends on the specific system / data for which the data-driven model is developed.

  * The model training is done with normalized values of inputs and outputs. Therefore, during model validation / simulation, the inputs are
    first normalized before subjected to the optimal models to generate the outputs. Subsequently, the normalized model outputs are converted
    back to the absolute (actual) scale and reported. Normalization is performed by the typical Max-Min Normalization approach.

  * Subsequent versions (updates) of the codes will include comparison and analyses with additional network architectures, as well as inclusion
    of additional objective functions while training the optimal NN model for a particular system, thus leading to sparse model identification.


