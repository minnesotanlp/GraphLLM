# Which Modality should I use -- Text, Motif, or Image? : Understanding Graphs with Large Language Models
This repository provides code of the following paper:

> [Which Modality should I use -- Text, Motif, or Image? : Understanding Graphs with Large Language Models](https://arxiv.org/abs/2311.09862) <br>
> [Debarati Das](https://debaratidas94.github.io/), Ishaan Gupta, Jaideep Srivasta, [Dongyeop Kang](https://dykang.github.io/) <br>

<p align="center" >
    <img src=assets/main_fig.jpg width=60%">
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
## Generating Graph samples and associated information
1. Sample the appropriate datasets with the graph sampling method, chooses N graphs and writes out the groundtruth, node with question mark, associated edgelist and y labels out. These are used to generate the text and motif modality. 
```
generate_graph_edgelist_ylabel.py 
```
2. Read the edgelists, and y labels from the previous step and generate the graph images. These are used to generate the image modality. 
```
generate_graph_images.py (old)
graphviz_image_generation.py (new)
```
## Graph Modality Prompting Scripts
Each script is run through the appropriate config.json file (present in config/ and associated python file). Running every script will produce a csv which calculates the avg and std of metrics across all runs, for all settings. We also log the specific prompts, responses and other associated outputs for each graph. 

1. Text Encoder
```
run_prompting_text_encoder.py
Associated config - config/config_textencoder.json

```
2. Image Encoder
```
run_prompting_image_encoder.py
Associated config - config/config_image_encoder.json

```
3. Motif Encoder
```
run_prompting_motif_encoder.py
Associated config - config/config_motif_encoder.json

```
4. Extending multiple modalities (combining 2 or more together)
```
run_prompting_textmotif_encoder.py
run_prompting_textimage_encoder.py
run_prompting_motifimage_encoder.py
run_prompting_allmodalities_encoder.py
```
5. Specific Modality experiments (depth first experiments)
```
all_text_modality_variations.py
all_motif_modality_variations.py
all_image_modality_variations.py
```
6. Experiments with few shot and few shot + rationale
```
run_few_shot_prompting.py
```


## Citation
If you find this work useful for your research, please cite our paper:

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