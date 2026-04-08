import os
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

# in pytohn class names are in CamelCase;
# non-class names (e.g. functions/variables) are in snake_case
class Chat:
    '''
    >>> chat = Chat()
    >>> chat.send_message('my name is bob', temperature=0.0)
    'Arrr, ye be Bob, eh? Yer name be known to me now, matey.'
    >>> chat.send_message('what is my name?', temperature=0.0)
    "Ye be askin' about yer own name, eh? Yer name be... Bob, matey!"

    >>> chat2 = Chat()
    >>> chat2.send_message('what is my name?', temperature=0.0)
    "Arrr, I be not aware o' yer name, matey. Yer identity be a mystery to me."
    '''
    client = Groq()
    def __init__(self):
        self.messages = [
                {
                    # most important content for sys prompt is length of response
                    "role": "system",
                    "content": "Write the output in 1-2 sentences. Talk like pirate."
                },
            ]
    def send_message(self, message, temperature=0.8):
        self.messages.append(
            {
                # system: never change; user: changes a lot;
                # the message that you are sending to the AI
                'role': 'user',
                'content': message
            }
        )
        # in order to make non-deterministic code deterministic;
        # in general very hard CS problem;
        # in this case, has a "temperature" param that controls randomness;
        # the higher the value, the more randomness;
        # hihgher temperature => more creativity
        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model="llama-3.1-8b-instant",
            temperature=temperature,
        )
        result = chat_completion.choices[0].message.content
        self.messages.append({
            'role': 'assistant',
            'content': result
        })
        return result


def repl():
    '''
>>> def monkey_input(prompt, user_inputs=['Hello, I am monkey.', 'Goodbye.']):
...     try:
...         user_input = user_inputs.pop(0)
...         print(f'{prompt}{user_input}')
...         return user_input
...     except IndexError:
...         raise KeyboardInterrupt
>>> import builtins
>>> builtins.input = monkey_input
>>> repl()
    chat> Hello, I am monkey.
    Arrr, a sneaky little monkey, eh? Ye be swingin' into our conversation, matey!
    chat> Goodbye.
    Arrr, a sneaky little monkey, eh? Ye be swingin' into our conversation, matey!
    <BLANKLINE> 
    '''
    import readline
    chat = Chat()
    try:
        while True:
            user_input = input('chat> ')
            response = chat.send_message(user_input)
            print(response)
    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    repl()