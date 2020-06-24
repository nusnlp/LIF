# LIF  #

This repository contains the data and source code of the
paper [Learning to Identify Follow-up Questions in Conversational
Question Answering](https://www.aclweb.org/anthology/2020.acl-main.90.pdf).


### Overview  ###

This work is focused on how a machine can learn to identify follow-up questions in
Conversational QA settings. It is crucial for a system to
be able to determine whether a question is
a follow-up question of the current conversation for effective answer finding.
In this work, we introduce a new follow-up question identification task.
We propose a three-way attentive pooling network
that determines the suitability of a follow-up
question by capturing pair-wise interactions
between the associated passage, the conversation history, and a candidate
follow-up question.


### Publication ###

If you use the data, source code or models from this work, please cite our paper:

```
@article{kundu2020lif,
  author    = {Kundu, Souvik and Lin, Qian and Ng, Hwee Tou},
  title     = {Learning to Identify Follow-up Questions in Conversational Question Answering},
  booktitle = {Proceedings of ACL},
  year      = {2020},
}
```


### Requirements ###

We use Python3.6 and AllenNlp. Install the packages listed in the `requirements.txt` file.


### Data ###

Run the `download_data.sh`:
```
sh download_data.sh
```

The data files will be downloaded inside `data/dataset`, and the embedding file will
be downloaded inside `data/embeddings`.


### Training a New Model ###

A new model (Three-way Attentive Pooling Network) can be trained from scratch
by executing the following example command:

```bash
allennlp train training_configs/l2af_3way_ap.json -s models/3way_ap --include-package l2af
```

Similarly, a new BERT-based baseline model can be trained using the following command:

```bash
allennlp train training_configs/bert_baseline.json -s models/bert_baseline --include-package l2af
```


### Prediction ###

You can predict for the test instances using the trained models
by running the following command:

```bash
allennlp predict models/3way_ap/model.tar.gz data/dataset/test_i.jsonl \
                    --output-file models/3way_ap/test_i_predictions.jsonl \
                    --batch-size 32 \
                    --silent \
                    --cuda-device 0 \
                    --predictor l2af_predictor_binary \
                    --include-package l2af
```

One can also download our pre-trained models and use it for prediction. The models can be downloaded
by running `download_pretrained_models.sh`:

```bash
sh download_pretrained_models.sh
```

This should download the models inside `data/pretrained-model`.


### Evaluation ###

For evaluation, you need to generate the prediction file for dev set and test set.
Running the prediction is necessary as we estimate the threshold based
on the performance on the dev set. To run the evaluation, simply run:

```bash
python evaluator.py --dev_pred_file /path/to/dev_predictions.jsonl \
                    --test_pred_file /path/to/test_predictions.jsonl
```
