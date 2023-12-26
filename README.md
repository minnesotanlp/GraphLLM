# Which Modality should I use -- Text, Motif, or Image? : Understanding Graphs with Large Language Models
This repository provides code of the following paper:

> [Which Modality should I use -- Text, Motif, or Image? : Understanding Graphs with Large Language Models](https://arxiv.org/abs/2311.09862) <br>
> [Debarati Das](https://debaratidas94.github.io/), Ishaan Gupta, Jaideep Srivasta, [Dongyeop Kang](https://dykang.github.io/) <br>

<p align="center" >
    <img src=assets/main_fig.jpg width="20%">
</p>


## Environment Setup
```
conda create --name graphllm python==3.10.12
conda activate graphllm
pip install torch
pip install openai
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install numpy pandas matplotlib seaborn scipy
```
## Scripts
```
filler
```

```
@misc{das2023modality,
      title={Which Modality should I use -- Text, Motif, or Image? : Understanding Graphs with Large Language Models}, 
      author={Debarati Das and Ishaan Gupta and Jaideep Srivastava and Dongyeop Kang},
      year={2023},
      eprint={2311.09862},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```