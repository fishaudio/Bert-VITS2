# bert-vits
vits with bert

put [pytorch_model.bin](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin)
to bert/roberta_wwm_ext_large/  
Then resample all audio to 44100Hz,generate filelists.  
Please read the code in preprocess_text.py to generate the specified format filelist.  
Run preprocess_text.py  
Run spec_gen.py  
Run train_ms.py(Only multi speaker training is supported)  
