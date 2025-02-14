<h3 align="center">
Our goal is bringing AI innovation to everyone with affordable hardwares.
</h3>

---
WuKong is an easy-to-use framework for LLM inference and agent serving in golang from scratch and try to solve the [issue](https://github.com/liuy/wukong/issues/1).

## Build from source
### Install dependency
```bash
sudo apt-get install cuda-toolkit # for cuda-runtime
sudo apt-get install golang # for golang tools
sudo apt-get install cmake make # for makefiles
```
### Configure cuda ENVs
It'd better to put these 'exports' in ~/.bashrc
```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export PATH=$PATH:$(go env GOPATH)/bin
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### Compile source
```bash
git clone git@github.com:liuy/wukong.git # get the source
cd wukong/
make # kick start compiling
```

### Pull and Run the models from internet
Right now we support pulling models from huggingface.co, modelscope.cn and ollama.com
```bash
./wk run hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF # Download and run model from huggingface.com
```
```bash
./wk run modelscope.cn/bartowski/Llama-3.2-1B-Instruct-GGUF # Download and run model from modelscople.cn of Alibaba
```
```bash
./wk run llama3.2:1b # Download and run model from ollama.com
```


