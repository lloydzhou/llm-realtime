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

