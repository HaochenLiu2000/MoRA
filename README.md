# Mixture-of-Experts for Knowledge Graph Retrieval-Augmented Generation

## 1. Datasets

The datasets are from [GraftNet](https://github.com/haitian-sun/GraftNet) and [KG-GPT](https://github.com/jiho283/KG-GPT/). 

Download [Webqsp](http://curtis.ml.cmu.edu/datasets/graftnet/data_webqsp.zip), [Wikimovies](http://curtis.ml.cmu.edu/datasets/graftnet/data_wikimovie.zip) and [MetaQA](https://github.com/yuyuz/MetaQA).

Place the files or folders `metaqa_kg.pickle`, `kb.txt`, `1-hop/vanilla`(renamed `1-hop/`), `2-hop/vanilla`(renamed `2-hop/`) under `./metaqa`.

Place the folder `webqsp` and `wikimovie` for the other two datasets.

Download the Llama2-chat model [here](https://huggingface.co/meta-llama) and place the model under `/MoRA/`.

## 2. Dependencies

Run the following commands to create a conda environment:

    conda create -y -n mora python=3.11
    conda activate mora
    pip install numpy==2.0.1
    pip install torch==2.4.0
    pip install transformers==4.46.2
    pip install tqdm
    pip install scikit-learn==1.5.1
    pip install openai
    pip install torch_geometric

Write your own OpenAI API key and Anthropic API key in the first few lines of `./test_llmqa.py`.

## 3. Setting up

Run

    python process_document.py

## 4. Training

Run

    python test.py

You can check and choose the dataset, model and phase in the first few lines of the code. For training, set the phase to `train`.


## 5. Evidence Generation

Run

    python test.py

You can check and choose the dataset, model and phase in the first few lines of the code. For Evidence Generation, set the phase to `test`.

## 6. Evaluating

For the final generation of question answering using retrieved evidence.

Run

    python test_llmqa.py

You can choose the dataset and evidence file name in the code. 

## 7. Acknowledgment

Our data is based on the code of GraftNet and KG-GPT. Thanks to the authors and developers!