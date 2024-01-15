FROM pytorch/pytorch:latest

RUN apt update -y
RUN apt upgrade -y
RUN pip install pandas scikit-learn tqdm transformers matplotlib sentencepiece

WORKDIR /PSTALN
COPY ./dataset.py ./
COPY ./transformer.py ./
COPY ./train.py ./
COPY ./train_BIGO.csv ./
COPY ./test_BIGO.csv ./
COPY ./val_BIGO.csv ./
COPY ./val_IGO.csv ./
COPY ./train_IGO.csv ./
COPY ./test_IGO.csv ./
COPY ./viterbi.py ./