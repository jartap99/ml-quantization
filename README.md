
- [Pytorch Quantization](#pytorch-quantization)
  - [References](#references)
  - [Key points / Limitations](#key-points--limitations)
- [Tensorflow Quantization](#tensorflow-quantization)
  - [References](#references-1)
  - [Key points / Limitations](#key-points--limitations-1)
- [Qualcomm - AIMET](#qualcomm---aimet)
  - [References](#references-2)
- [NVIDIA Toolkits](#nvidia-toolkits)
  - [Tensorflow](#tensorflow)
  - [Pytorch](#pytorch)

# Pytorch Quantization

## References

1. [Torch quantization design proposal](https://github.com/pytorch/pytorch/wiki/torch_quantization_design_proposal)

2. Refer to the github link


## Key points / Limitations

1. ...

# Tensorflow Quantization

![TF Quant](./tf_quant.png)

## References

1. [Model optimization toolkit](https://www.tensorflow.org/lite/performance/model_optimization)

2. [QAT](https://www.tensorflow.org/model_optimization/guide/quantization/training)

3. [QAT Example](https://www.tensorflow.org/model_optimization/guide/quantization/training_example)

4. [Google's quantization paper - 2017](https://arxiv.org/pdf/1712.05877.pdf) 


## Key points / Limitations

1. ...
   
# Qualcomm - AIMET

## References

1. [AIMET Github](https://github.com/quic/aimet)

2. [AIMET - Github pages](https://quic.github.io/aimet-pages/index.html)

3. [AIMET User guide](https://quic.github.io/aimet-pages/releases/latest/user_guide/index.html) 


# NVIDIA Toolkits

NVIDIA has QAT toolkit for TF2 with goal of accelerating quantized networks with NVDIIA TensorRT on NVIDIA GPUs.

[Accelerating Quantized Networks with the NVIDIA QAT Toolkit for TensorFlow and NVIDIA TensorRT - June 2022](https://developer.nvidia.com/blog/accelerating-quantized-networks-with-qat-toolkit-and-tensorrt/)

[Quantization basics](https://arxiv.org/pdf/2004.09602.pdf)

Accompanied by this [Toward INT8 Inference: An End-to-End Workflow for Deploying Quantization-Aware Trained Networks Using TensorRT](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41440/)

[Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training with NVIDIA TensorRT](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)

[TENSORT FUSING, ETC - User guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#qdq-placement-recs) 

## Tensorflow

1. [Github - NVIDIA TF 2 quantization](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization)

2. [Tensorflow-quantization userguide](https://docs.nvidia.com/deeplearning/tensorrt/tensorflow-quantization-toolkit/docs/index.html)


3. Features/limitations
   
    a. Only QAT is supported. 

    b. int16, fp16 are supported


## Pytorch

1. [Github - NVIDIA pytorch quantization](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization)

2. [Pytorch-quantization userguide](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html)

3. They acknowledge QAT is not a solved problem mathematically (discrete numerical optimization problem). Based on their experience, they recommend the following

    a. Use small learning rate for STE (Straight Through Estimater) to wokr well. 

    b. Do not change quantization representation (scale) during training (atleast not frequenclt). 