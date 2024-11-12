"""
realtime api
1. session.created
2. session.update
3. session.updated
4. input_audio_buffer.append
4. input_audio_buffer.speech_started
4. input_audio_buffer.speech_stopped
4. input_audio_buffer.committed
4. conversation.item.input_audio_transcription.completed  //
4. conversation.item.created(user)
5. response.created
6. response.output_item.added
7. conversation.item.created(assistant)
8. response.content_part.added
9. response.audio_transcript.delta / response.text.delta
10. response.audio.delta
9. response.audio_transcript.done / response.text.done
10. response.audio.done
8. response.content_part.done
7. response.output_item.done
6. response.done
"""
import os
import base64
import json
import numpy
import fsmnvad
import asyncio
import logging
import tornado.log
import tornado.ioloop
import tornado.web
import tornado.websocket
from functools import partial
from nanoid import generate as nanoid
from litellm import completion
from pathlib import Path


root_dir = Path(os.path.dirname(os.path.abspath(__file__)))


class RealtimeHandler(tornado.websocket.WebSocketHandler):

    def initialize(self):
        logging.info("initialize %r %r %r", self, self.request, self.request.headers)
        self.session = dict(
            id=nanoid(),
            object='realtime.session',
            model=self.get_argument('model', 'gpt-4o-mini'),
            modalities=['text', 'audio'],
            instructions='',
            voice='alloy',
            input_audio_format='pcm16',
            output_audio_format='pcm16',
            input_audio_transcription={'model': 'whisper-1'},
            turn_detection=dict(
                type='server_vad',
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=500,
            ),
            tools=[],
            tool_choice='auto',
            temperature=0.8,
            max_response_output_tokens=None,
        )
        self.conversation = list()
        self.response_task = None
        self.transcript_task = None
        self.input_audio_buffer = numpy.array([])
        self.new_user_message_id = None
        self.audio_end_ms = 0
        self.segments_result = [[-1, 0]]
        self.segments_cache = []
        self.vad_online = fsmnvad.FSMNVadOnline(root_dir / 'config.yaml')
        # OpenAI realtime API sample_rate=24000, but fsmnvad using sample_rate=16000
        # frontend = self.vad_online.frontend
        # frontend.opts.frame_opts.samp_freq = 24000
        # frontend.frame_sample_length = int(frontend.opts.frame_opts.frame_length_ms * frontend.opts.frame_opts.samp_freq / 1000)
        # frontend.frame_shift_sample_length = int(frontend.opts.frame_opts.frame_shift_ms * frontend.opts.frame_opts.samp_freq / 1000)
        
    def check_origin(self, origin):
        return True

    def select_subprotocol(self, subprotocols):
        logging.info("select_subprotocol %r", subprotocols)
        return subprotocols[0]

    def open(self):
        logging.info("open websocket %r", self.request)
        self.server_event('session.created', session=self.session)

    def on_message(self, message):
        logging.info("on_message %r", message[:100])
        try:
            event = json.loads(message)
            self.client_event(**event)
        except Exception as e:
            logging.exception(e)
            logging.error(e)

    def client_event(self, id='', type='', **kwargs):
        if type == 'conversation.item.create':
            self.create_conversation_item(item=kwargs['item'])
        elif type == 'response.create':
            self.cancel_task()
            self.response_task = asyncio.create_task(self.create_response())
        elif type == 'session.update':
            # TODO
            self.server_event('session.updated', session=self.session)
        elif type == 'input_audio_buffer.append':
            self.append_audio_buffer(**kwargs)

    def append_audio_buffer(self, audio='', **kwargs):
        audio_data = numpy.frombuffer(
            base64.b64decode(audio.encode()),
            numpy.int16
        ).flatten().astype(numpy.float32) / (1 << 15)
        self.input_audio_buffer = numpy.append(self.input_audio_buffer, audio_data)
        segments_result, self.segments_cache = self.vad_online.segments_online(
            audio_data,
            in_cache=self.segments_cache,
            is_final=False,
        )
        if segments_result:
            audio_start_ms = segments_result[-1][0]
            if audio_start_ms != self.segments_result[-1][0]:
                self.new_user_message_id = nanoid()
                self.server_event('input_audio_buffer.speech_started', audio_start_ms=audio_start_ms, item_id=self.new_user_message_id)
            self.segments_result = segments_result
        else:
            audio_end_ms = self.segments_result[-1][1]
            if audio_end_ms and audio_end_ms != self.audio_end_ms:
                self.audio_end_ms = audio_end_ms
                silence_duration_ms = self.session.get('turn_detection', {}).get('silence_duration_ms', 500)
                sample_rate = self.vad_online.frontend.opts.frame_opts.samp_freq
                duration = len(self.input_audio_buffer) / sample_rate * 1000
                if duration - silence_duration_ms > audio_end_ms:
                    item_id = self.new_user_message_id
                    self.server_event(
                        'input_audio_buffer.speech_stopped',
                        audio_end_ms=audio_end_ms,
                        item_id=item_id,
                    )
                    previous_item_id = self.conversation[-1].get('id') if len(self.conversation) > 0 else None
                    self.server_event(
                        'input_audio_buffer.committed',
                        audio_end_ms=audio_end_ms,
                        item_id=item_id,
                        previous_item_id=previous_item_id,
                    )
                    self.cancel_task()
                    self.transcript_task = asyncio.create_task(self.create_transcript(item_id, previous_item_id))

    async def create_transcript(self, item_id, previous_item_id):
        # TODO using whisper to get transcript
        await asyncio.sleep(1)
        transcript = 'Hi'
        self.create_conversation_item(item={
            'id': item_id,
            'type': 'message',
            'role': 'user',
            'content': [{
                'type': 'input_audio',
                'transcript': None,
            }]
        }, previous_item_id=previous_item_id)
        # 在openai的示例里面这个事件是在后面触发的
        self.update_conversation_item(item_id, content=[{
            'type': 'input_audio',
            'transcript': transcript
        }])
        self.server_event(
            'conversation.item.input_audio_transcription.completed',
            item_id=item_id,
            transcript=transcript,
            content_index=0,
        )
        self.cancel_task()
        self.response_task = asyncio.create_task(self.create_response())

    def cancel_task(self, send_event=True):
        if self.transcript_task:
            self.transcript_task.cancel()  # 取消旧的任务
            self.transcript_task = None
        if self.response_task:
            self.response_task.cancel()  # 取消旧的任务
            self.response_task = None
            item_id = self.conversation[-1].get('id')
            logging.warn("cancel_task %r", item_id)
            if send_event:
                self.server_event(
                    'conversation.item.truncated',
                    item_id=item_id, content_index=0, audio_end_ms=0
                )

    def create_conversation_item(self, item=None, status='completed', previous_item_id=None, **kwargs):
        item.update(status=status, object='realtime.item')
        self.conversation.append(item)
        self.server_event('conversation.item.created', item=item, previous_item_id=previous_item_id)

    def update_conversation_item(self, item_id, content=None):
        for item in self.conversation:
            if item_id == item['id']:
                item['content'] = content

    async def create_response(self, type='text'):
        """
        5. response.created
        6. response.output_item.added
        7. conversation.item.created(assistant)
        8. response.content_part.added
        9. response.audio_transcript.delta  / response.text.delta
        10. response.audio.delta
        9. response.audio_transcript.done / response.text.done
        10. response.audio.done
        8. response.content_part.done
        7. response.output_item.done
        6. response.done
        """
        response_id = nanoid()
        self.server_event('response.created', response={
            'id': response_id,
            'object': 'realtime.response',
            'status': 'in_progress',
            'status_details': None,
            'output': [],
            'usage': None,
        })
        item_id = nanoid()
        self.server_event('response.output_item.added', item={
            'id': item_id,
            'object': 'realtime.item',
            'type': 'message',
            'status': 'in_progress',
            'role': 'assistant',
            'content': [],
        }, response_id=response_id, output_index=0)

        previous_item_id = self.conversation[-1].get('id')
        self.create_conversation_item(item={
            'id': item_id,
            'type': 'message',
            'role': 'assistant',
            'content': []
        }, status='in_progress', previous_item_id=previous_item_id)

        self.server_event('response.content_part.added', part={
            'type': type,
            'transcript': '', 'text': '',
        }, response_id=response_id, item_id=item_id, output_index=0, content_index=0)
        # get delta from llm
        content = []
        async for delta in self.llm():
            # mock get response
            content.append(delta)
            self.server_event(
                'response.audio_transcript.delta' if type == 'audio' else 'response.text.delta',
                delta=delta,
                response_id=response_id,
                item_id=item_id,
                output_index=0,
                content_index=0
            )

        text = ''.join(content)
        self.update_conversation_item(item_id, content=[{
            'type': type,
            'transcript': text, 'text': text,
        }])
        self.server_event(
            'response.audio_transcript.done' if type == 'audio' else 'response.text.done',
            transcript=text, text=text,
            response_id=response_id,
            item_id=item_id,
            output_index=0,
            content_index=0,
        )
        self.server_event('response.content_part.done', part={
            'type': type,
            'transcript': text, 'text': text,
        }, response_id=response_id, item_id=item_id, output_index=0, content_index=0)
        self.server_event('response.output_item.done', item={
            'id': item_id,
            'object': 'realtime.item',
            'type': 'message',
            'status': 'completed',
            'role': 'assistant',
            'content': [{
                'type': type,
                'transcript': text, 'text': text,
            }],
        }, response_id=response_id, output_index=0)
        self.server_event('response.done', response={
            'id': response_id,
            'object': 'realtime.response',
            'status': 'completed',
            'status_details': None,
            'output': [{
                'type': item_id,
                'object': 'realtime.item',
                'type': 'message',
                'status': 'completed',
                'role': 'assistant',
                'content': [{
                    'type': type,
                    'transcript': text,
                    'text': text,
                }]
            }],
        }, usage={
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'input_token_details': {
                'cached_tokens': 0,
                'text_tokens': 0,
                'audio_tokens': 0,
                'cached_tokens_details': {
                    'text_tokens': 0,
                    'audio_tokens': 0
                }
            },
            'output_token_details': {
                'text_tokens': 0,
                'audio_tokens': 0
            },
        })
        self.response_task = None

    async def llm(self):
        import litellm
        litellm.set_verbose = True
        model = self.session.get('model')
        if model == 'gpt-4o-realtime-preview-2024-10-01':
            model = 'gpt-4o-mini'  # TODO
        messages = [{
            'role': item['role'],
            'content': item['content'][0].get('text') or item['content'][0].get('input_text') or item['content'][0].get('transcript') ,
        } for item in self.conversation if len(item['content']) > 0]
        response = completion(model=model, messages=messages, stream=True)
        for part in response:
            logging.info("part %r", part)
            yield part.choices[0].delta.content or ""

    def server_event(self, event_type, **kwargs):
        data = dict(
            type=event_type,
            event_id=nanoid(),
        )
        data.update(kwargs)
        self.write_message(json.dumps(data))

    def on_close(self):
        logging.info("close websocket %r", self.request)
        self.cancel_task(False)

def main():
    tornado.log.enable_pretty_logging()
    # Create a web app whose only endpoint is a WebSocket, and start the web
    # app on port 8888.
    app = tornado.web.Application(
        [(r"/realtime", RealtimeHandler)],
        websocket_ping_interval=10,
        websocket_ping_timeout=30,
        debug=True,
    )
    app.listen(8888)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
