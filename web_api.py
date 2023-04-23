########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os
import copy
import types
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import trange
import uvicorn


app = FastAPI()
args = types.SimpleNamespace()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


current_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    current_path,
    'model',
    "RWKV-4-Novel-3B-v1-Chn-20230412-ctx4096"
)

# args.strategy = 'cuda fp16'
args.strategy = 'cuda fp16'
os.environ["RWKV_JIT_ON"] = '1'
# '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
os.environ["RWKV_CUDA_ON"] = '0'
CHAT_LANG = 'Chinese'  # English // Chinese // more to come
args.MODEL_NAME = model_path

# -1.py for [User & Bot] (Q&A) prompt
# -2.py for [Bob & Alice] (chat) prompt
# CHAT_LEN_SHORT = 40
# CHAT_LEN_LONG = 150
# FREE_GEN_LEN = 256
AVOID_REPEAT = '，：？！'
# 将一句话切成若干片段，每个片段不超过 MAX_CHUNK_LEN 个字，以避免显存不足，不过片段太短会降低速度，而且会影响聊天质量
# CHUNK_LEN = 256  # split input into chunks to save VRAM (shorter -> slower)

# For better chat & QA quality: reduce temp, reduce top-p, increase repetition penalties
# Explanation: https://platform.openai.com/docs/api-reference/parameter-details
GEN_TEMP = 1.1  # It could be a good idea to increase temp when top_p is low
GEN_TOP_P = 0.7  # Reduce top_p (to 0.5, 0.2, 0.1 etc.) for better Q&A accuracy (and less diversity)
GEN_alpha_presence = 0.2  # Presence Penalty
GEN_alpha_frequency = 0.2  # Frequency Penalty
###############################################################################

# Load Model

print(f'Loading model - {args.MODEL_NAME}')
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
pipeline = PIPELINE(model, f"{current_path}/20B_tokenizer.json")
END_OF_TEXT = 0
END_OF_LINE = 187
# pipeline = PIPELINE(model, "cl100k_base")
# END_OF_TEXT = 100257
# END_OF_LINE = 198

model_tokens = []
model_state = None

# 获取避免重复的标点信息
AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = pipeline.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd
################################################################################

def run_rnn(tokens, chunk_len, newline_adj = 0):
    global model_tokens, model_state
    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    # print(f'### model ###\n{tokens}\n[{pipeline.decode(model_tokens)}]')
    # model state获取4个每层的attention信息+一个FFN信息，每次获取的信息都是上一次的输出
    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:chunk_len], model_state)
        tokens = tokens[chunk_len:]

    out[END_OF_LINE] += newline_adj # adjust \n probability
    # 去多余符号
    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    return out

all_state = {} # 又一个全局变量
def save_all_state(srv, name, last_out):
    """
    将当前的状态保存到all_state中，以便后续恢复
    Args:
        srv (_type_): 分支
        name (_type_): 储存名
        last_out (_type_): 上一次输出结果
    """
    key = f'{name}_{srv}'
    all_state[key] = {}
    all_state[key]['out'] = last_out
    all_state[key]['state'] = copy.deepcopy(model_state)
    all_state[key]['token'] = copy.deepcopy(model_tokens)


def load_all_state(srv, name):
    """
    加载状态
    Args:
        srv (_type_): 分支
        name (_type_): 储存名

    Returns:
        _type_: _description_
    """
    global model_tokens, model_state
    n = f'{name}_{srv}'
    model_state = copy.deepcopy(all_state[n]['state'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']

################################################################################


def on_message(message, free_gen_len, chunk_len):
    global model_tokens, model_state

    srv = 'dummy_server'

    msg = message.replace('\\n','\n').strip()

    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P
    if "-temp=" in msg:
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp="+f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if "-top_p=" in msg:
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p="+f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    x_temp = min(max(0, 2, x_temp), 5)
    x_top_p = max(0, x_top_p)
    msg = msg.strip()
    if msg == '+reset':
        out = load_all_state('', 'chat_init')
        save_all_state(srv, 'chat', out)
        return
    elif msg[:5].lower() == '+gen ':
        new = '\n' + msg[5:].strip()
        # 如果是+gen，那么就是生成模式，所以需要将之前的状态清空，代表首次生成
        model_state = None
        model_tokens = []
        out = run_rnn(pipeline.encode(new), chunk_len)
        # gen_0 代表生成模式的初始状态,一般也是存输入端的状态
        save_all_state(srv, 'gen_0', out)

    elif msg.lower() == '+++':
        try:
            out = load_all_state(srv, 'gen_1')
            save_all_state(srv, 'gen_0', out)
        except Exception as err:
            print(err)
            return

    elif msg.lower() == '++':
        try:
            out = load_all_state(srv, 'gen_0')
        except Exception as err:
            print(err)
            return

    begin = len(model_tokens)
    out_last = begin
    occurrence = {}
    result_msg = ""
    for i in trange(free_gen_len+100):
        for n in occurrence:
            out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
        token = pipeline.sample_logits(
            out,
            temperature=x_temp,
            top_p=x_top_p,
        )
        if token == END_OF_TEXT:
            break
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        if msg[:4].lower() == '+qa ': # or msg[:4].lower() == '+qq ':
            out = run_rnn([token], chunk_len, newline_adj=-2)
        else:
            out = run_rnn([token], chunk_len,)
        # 将token反序列化为字符串,并且当字符串中不包含\ufffd时才进行输出，长度大于free_gen_len时停止输出
        temp_out = pipeline.decode(model_tokens[out_last:])
        if '\ufffd' not in temp_out: # avoid utf-8 display issues
            # 这里已经是输出了，弄到终端去
            # print(xxx, end='', flush=True)
            result_msg += temp_out
            out_last = begin + i + 1
            if i >= free_gen_len:
                break
    # send_msg = pipeline.decode(model_tokens[begin:]).strip()
    # print(f'### send ###\n[{send_msg}]')
    # reply_msg(send_msg)
    # 保存回复状态
    save_all_state(srv, 'gen_1', out)
    data = {
        "label": result_msg,
        "kind": 0,
        "insertText": result_msg,
        "detail": result_msg
    }
    return [data]
    # data_list = []
    # for text in result_msg.split():
    #     data = {
    #         "label": text,
    #         "kind": 0,
    #         "insertText": text,
    #         "detail": text
    #     }
    #     data_list.append(data)
    # return data_list


########################################################################################################
class Data(BaseModel):
    msg: str
    free_len: int = 256
    chunk_len: int = 512


@app.post("/gen/")
def gen(data: Data):
    return on_message(data.msg, data.free_len, data.chunk_len)


if __name__ == '__main__':
    uvicorn.run(
        app='web_api:app', host="127.0.0.1", port=6288, reload=True, workers=1,
    )


