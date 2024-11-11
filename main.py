"""
realtime api
1. session.created
2. session.update
3. session.updated
4. input_audio_buffer.append
4. input_audio_buffer.speech_started
4. input_audio_buffer.speech_stopped
4. input_audio_buffer.committed
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
import json
import asyncio
import logging
import tornado.log
import tornado.ioloop
import tornado.web
import tornado.websocket
from functools import partial
from nanoid import generate as nanoid


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
            input_audio_transcription=None,
            turn_detection=dict(
                type='server_vad',
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=200
            ),
            tools=[],
            tool_choice='auto',
            temperature=0.8,
            max_response_output_tokens=None,
        )
        self.conversation = list()
        self.response_task = None
        
    def check_origin(self, origin):
        return True

    def select_subprotocol(self, subprotocols):
        logging.info("select_subprotocol %r", subprotocols)
        return subprotocols[0]

    def open(self):
        logging.info("open websocket %r", self.request)
        self.server_event('session.created', session=self.session)

    def on_message(self, message):
        logging.info("on_message %r", message)
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

    def cancel_task(self, send_event=True):
        if self.response_task:
            item_id = self.conversation[-1].get('id')
            self.response_task.cancel()  # 取消旧的任务
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

    async def llm(self):
        # TODO
        for delta in ['Hello! ', 'How ', 'can I ', 'assist ', 'you ', 'today?']:
            await asyncio.sleep(0.1)
            yield delta

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
