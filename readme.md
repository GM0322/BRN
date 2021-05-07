# BRN
Codebase for "A block reconstruction network (BRN) approach for computed tomography"

## Preparation
Prepare python environment using Anacoda3 and install follow package
* pythorch
* numpy 
* os
* astra

## Evaluation Demo
**We provide a trained model for demonstration**

First, Unzip the saved model file to the './checkpoints' folder.
Then, set parameters and run the following code:

```Shell
python test.py
```

**In addation, considering that the python version runs slowly, we open the c++ source of test process in 'cpp' folder. The average time is about 0.09s for 50 times, and only 0.05s for stable running.**

Note: Need to convert the saved model to traced script module which supported by libtorch. you can run the code of 'model2traced_script_module.py'


## trainning 
If you want to re-train the network parameters, you can train the pre-trainning processing and trainning processing as the following two commands. 
```Shell
python pretrain.py
```

```Shell
python train.py
```

## Contact
For any question, please file an issue or contact
```
Genwei Ma: magenwei@126.com
```
