# PDEnsorflow 

**PDEnsorflow**  is a library developed under `TensorFlow 2.X` to solve Partial dfferential equations


## Pre-requisites
Install `TensorFlow 2.X` using anaconda. First create an environment and activate it; e.g.: 

```
conda create --name tf_gpu
conda activate tf_gpu
```

Then, follow the instructions to install tensorflow [here](https://www.tensorflow.org/install/pip).


## Run the code

Once the environment is activated, source the `init.sh` file in the main directory of PDEnsorflow:

```
source init.sh
```

Then, launch one of the examples; e.g.:

```
cd Tests/Fenton_atria
python fenton.py
```


**Note**: *This run **PDEnsorflow** under GPU, provided that libraries are correctly installed. Otherwise, it will run under standard CPU. 
In the examples, the console will show under wich device the code is executed.
