# chainer-PointerSentinelMixtureModels
Chainer implementation of Pointer Sentinel Mixture Models(https://arxiv.org/abs/1609.07843)

## Requirement
Chainer v1.15.0.1

## Running
### Training data format
You can prepare traing data as `train_data.txt`. In `train_data.txt`, every line has one JSON which has input data and output data represented as list.  
This means following format
```
{"input": ["I", "have", "a", "pen", ".", "I", "have", "an", "apple", "."], "output": ["apple", "pen"]}
{"input": ["I", "have", "a", "pen", ".", "I", "have", "a", "pineapple", "."], "output": ["pineapple", "pen"]}
...
```
And you must put `train_data.txt` on current directory.

### Traing
`PointerSentinelMixtureModels.py` contains code for building model.  
You can start training like following command:
```shell
$ python PointerSentinelMixtureModels.py --batchsize=64 --gpu=0 --embed=128 --unit=256 --out=result -L=30 --mode=train
```

### Restart training
If you want to restart traing by saves model, switch `--mode` parameter as `--mode=restart`,and add parameters for model and vocab, id2wd files, `--model` `--vocab_path` `--id_path`.  
You can restart training like following command:
```shell
$ python  PointerSentinelMixtureModels.py --batchsize=64 --gpu=0 --embed=128 --unit=256 --out=result -L=30 --mode=restart --vocab_path=data/vocab --id_path=data/id2wd --model=result/PointerSentinelMixtureModels-2000.model
```
Meaing of parameters are written in `PointerSentinelMixtureModels.py` file. Please read once.

### Decoding
After training, you can start decoding. In this code, I used beam search algorithm(this is not refered in paper).  
You can start decoding like following command:
```shell
$ python  PointerSentinelMixtureModels.py --embed=128 --unit=256  -L=30 --mode=decode --vocab_path=data/vocab --id_path=data/id2wd --model=result/PointerSentinelMixtureModels-500.model --data_path=train_data.txt --decode_path=decode.txt --beam_size=8 --max_length=30
```
`--data_path` is a parameter to specify file for encoding.  
Encoding file is required JSON format which has `input` parameter that have array object in every line.  

### Training log
In code, information about training(epoch, barchsize, batch loss) is recorded in log file `train.log`. 
```
INFO:root:Learning Start!
INFO:root:epoch: 1, batchsize: 64, loss: 4412.14013671875
INFO:root:epoch 1's calc time 32.55013418197632
INFO:root:epoch: 2, batchsize: 64, loss: 4932.1435546875
INFO:root:epoch 2's calc time 40.78377103805542
...
```

## Reference
Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher, "Pointer Sentinel Mixture Models" [arXiv:1609.07843](https://arxiv.org/abs/1609.07843)
