# Resistor_Identifier
 
The full post can be found on [my website](https://tylerebowers.com/posts/resistor_ml).

The pipeline for the identification task is as follows:

1. Identify where in the image the resistors are located. (Detectron 2)
2. Extract the color bands from each of the identified resistors. (Resnet18)
3. Decode the color bands to get the resistance value using an objective function. (Simple if/elif)

This project also uses a partially synthetic dataset. [Github](https://github.com/tylerebowers/synthetic_resistor_generation), [Huggingface](https://huggingface.co/datasets/tylerebowers/synthetic_resistors).
