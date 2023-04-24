### chatRWKV project introduction
[link](https://github.com/BlinkDL/ChatRWKV)


### Model Download
[link](https://huggingface.co/BlinkDL/rwkv-4-novel/tree/main)
- you can use wget to download it

### RWKV-Copilot extension for VSCode
[link](https://github.com/Tlntin/rwkv-copilot)

### how to run?
- install nvidia-driver, nvidia-cuda and python
```bash
pip install -r requirements.txt
uvicorn web_api:app --host 127.0.0.1  --port 6288 --reload --workers 1
```