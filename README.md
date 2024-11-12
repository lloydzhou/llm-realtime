# llm-realtime

## start server

```
pip install tornado nanoid litellm 

export OPENAI_API_KEY=xxxx
python main.py
```

## using Azure-Samples/aoai-realtime-audio-sdk

```
Azure-Samples/aoai-realtime-audio-sdk/tree/main
Azure-Samples/aoai-realtime-audio-sdk/tree/main

clientRef.current = new RTClient(
  {key: ''},
  {model: 'gpt-4o-mini', endpoint: new URL('ws://localhost:8888/realtime')},
)
```


## TODO

#### 语音输入
1. 通过端点检测[fsmn-vad](https://github.com/lovemefan/fsmn-vad)拿到开始时间和结束时间
2. 使用两个时间调用语音识别(whisper/whisper.cpp/SenseVoice/PaddleSpeech)结合时间拿到文本
3. 拿到文本之后调用LLM API

#### 语音输出
1. 传统tts/CosyVoice/...


