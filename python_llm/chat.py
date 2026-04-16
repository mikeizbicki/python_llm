"""AI-powered document chat agent.

Lets users query files using natural language and shell-like commands.
"""
import ast
import glob
import json
import operator
import os
import re
import sys
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

MODEL = 'openai/gpt-oss-120b'

CALCULATE_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'calculate',
        'description': (
            'Evaluate a simple arithmetic expression and '
            'return the result.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'expression': {
                    'type': 'string',
                    'description': (
                        'The arithmetic expression to evaluate, '
                        'e.g. "2 + 2" or "10 * (3 + 4)".'
                    ),
                },
            },
            'required': ['expression'],
        },
    },
}

LS_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'ls',
        'description': (
            'List files and folders in a directory. '
            'Optionally takes a path argument.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': (
                        'The directory path to list. Defaults to '
                        'the current directory.'
                    ),
                },
            },
            'required': [],
        },
    },
}

CAT_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'cat',
        'description': 'Read and return the contents of a text file.',
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'The path to the file to read.',
                },
            },
            'required': ['path'],
        },
    },
}

GREP_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'grep',
        'description': (
            'Search for lines matching a regex pattern '
            'in a file or directory.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'pattern': {
                    'type': 'string',
                    'description': 'The regex pattern to search for.',
                },
                'path': {
                    'type': 'string',
                    'description': (
                        'The file or directory path to search. '
                        'Defaults to current directory.'
                    ),
                },
            },
            'required': ['pattern'],
        },
    },
}

ALL_TOOL_SCHEMAS = [CALCULATE_SCHEMA, LS_SCHEMA, CAT_SCHEMA, GREP_SCHEMA]

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _eval_node(node):
    """
    Recursively evaluate a single AST node.

    Raises ValueError for unsafe expressions.

    >>> _eval_node(ast.parse('2 + 2', mode='eval').body)
    4
    >>> _eval_node(ast.parse('3 * 7', mode='eval').body)
    21
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError('invalid expression')
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError('invalid expression')
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_OPS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError('invalid expression')
        operand = _eval_node(node.operand)
        return _ALLOWED_OPS[op_type](operand)
    else:
        raise ValueError('invalid expression')


def is_path_safe(path):
    """
    Returns True if a path is safe to read.

    Checks for absolute paths or directory traversal.

    >>> is_path_safe('README.md')
    True
    >>> is_path_safe('chat.py')
    True
    >>> is_path_safe('/etc/passwd')
    False
    >>> is_path_safe('../secret.txt')
    False
    >>> is_path_safe('some/../file.txt')
    False
    >>> is_path_safe('.')
    True
    >>> is_path_safe('')
    True
    """
    if path.startswith('/'):
        return False
    parts = path.replace('\\', '/').split('/')
    if '..' in parts:
        return False
    return True

# in pytohn class names are in CamelCase;
# non-class names (e.g. functions/variables) are in snake_case
class Chat:
    '''
    A chat agent that talks with an LLM and helsp with tool usage.
    The Chat class stores talking history and allows messages to be sent
    to an LLM. It also supports tool calling (ls, cat, grep, calculate)
    by structured tool definitions.

    >>> chat = Chat()
    >>> chat.send_message('my name is bob', temperature=0.0)
    'Nice to meet you, Bob! How can I assist you today?'
    >>> chat.send_message('what is my name? just say my name', temperature=0.0)
    'Your name is Bob.'

    >>> chat2 = Chat()
    >>> chat2.send_message('what is my name?', temperature=0.0)
    'I don’t have any information about your name. If you’d like me to address you a certain way, just let me know!'
    '''
    client = Groq()

    def __init__(self):
        """Initializes the chat history with a system prompt that enforces a pirate persona."""
        self.client = Groq()
        self.messages = []
        self.tool_dispatch = {
            'calculate' : self.calculate,
            'ls': self.ls,
            'cat':self.cat,
            'grep':self.grep
        }
    
    def calculate(self, expression):
        """
        Evaluate a simple arithmetic expression and return the result.

        >>> c = Chat()
        >>> c.calculate('2 + 2')              # Hits the standard integer path
        '4'
        >>> c.calculate('100 / 4')            # Hits Branch: float.is_integer() with '/'
        '25.0'
        >>> c.calculate('5 * 5.0')            # Hits Branch: float.is_integer() without '/'
        '25'
        >>> c.calculate('10 / 3')             # Hits Branch: standard repeating float
        '3.3333333333333335'
        >>> c.calculate('1 / 0')              # Hits ZeroDivisionError branch
        'Error: division by zero'
        >>> c.calculate('1 + (2 *')           # Hits SyntaxError branch
        'Error: invalid expression'
        >>> c.calculate('__import__("os")')   # Hits ValueError branch via _eval_node
        'Error: invalid expression'
        >>> c.calculate('None + 1')           # Hits TypeError branch
        'Error: invalid expression'
        """
        try:
            tree = ast.parse(expression, mode='eval')
            result = _eval_node(tree.body)
            # Logic to handle how floats are displayed
            if isinstance(result, float) and result.is_integer():
                if '/' in expression and '//' not in expression:
                    return str(result)  # Keeps '.0' for standard division
                return str(int(result)) # Converts to int for other operations
            return str(result)
        except ZeroDivisionError:
            return 'Error: division by zero'
        except (ValueError, TypeError):
            return 'Error: invalid expression'
        except SyntaxError:
            return 'Error: invalid expression'
    
    def ls(self, folder="."):
        """
        List files/folders in a directory, asciibetically, one per line.

        >>> c = Chat()
        >>> 'python_llm/chat.py' in c.ls('.')
        False
        >>> c.ls('/etc')
        'Error: unsafe path'
        >>> c.ls('../other')
        'Error: unsafe path'
        >>> c.ls('nonexistent_folder_xyz')
        ''
        """
        
        if not is_path_safe(folder):
            return 'Error: unsafe path'
        files = sorted(glob.glob(f'{folder}/*'))
        names = [os.path.basename(f) for f in files]
        return '\n'.join(names)
        
    def cat(self, path):
        '''
        Opens a file and returns its contents as a string.

        >>> c = Chat()
        >>> c.cat('python_llm/chat.py')[1:11]
        '\"\"AI-power'
        >>> c.cat('nonexistent_file_xyz.txt')
        'Error: file not found'
        >>> c.cat('/etc/passwd')
        'Error: unsafe path'
        >>> c.cat('../secret.txt')
        'Error: unsafe path'

        '''
        if not is_path_safe(path):
            return 'Error: unsafe path'
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return 'Error: file not found'
        except UnicodeDecodeError:
            try:
                with open(path, 'r', encoding='utf-16') as f:
                    return f.read()
            except Exception:
                return 'Error: cannot decode file'
        except Exception as e:
            return f'Error: {e}'
    

    def grep(self, pattern, path):
        """
        Search for lines matching a regex pattern (recursive).
        Returns matching lines as 'filename:line', or an error string

        >>> c = Chat()
        >>> result = c.grep('def is_path_safe', 'python_llm/chat.py')
        >>> 'python_llm/chat.py' in result
        True
        >>> c.grep('def ', '/etc')
        'Error: unsafe path'
        >>> c.grep('def ', '../other')
        'Error: unsafe path'
        >>> c.grep('zzz[n]omatch_xyz', 'chat.py')
        ''
        >>> c.grep('[invalid', 'chat.py')
        'Error: invalid pattern: unterminated character set at position 0'
        """
        if not is_path_safe(path):
            return 'Error: unsafe path'
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return f'Error: invalid pattern: {e}'

        results = []
        if os.path.isfile(path):
            files = [path]
        else:
            files = []
            for root, dirs, filenames in os.walk(path):
                dirs[:] = sorted([d for d in dirs if not d.startswith('.')])
                for fname in sorted(filenames):
                    files.append(os.path.join(root, fname))

        for filepath in files:
            try:
                with open(
                    filepath, 'r', encoding='utf-8', errors='ignore'
                ) as f:
                    for line in f:
                        if compiled.search(line):
                            results.append(f'{filepath}:{line.rstrip()}')
            except Exception:
                continue

        return '\n'.join(results)
    
    def run_tool(self, name, args):
        """Dispatches a tool call by name with args"""
        return self.tool_dispatch[name(**args)]


        # in order to make non-deterministic code deterministic;
        # in general very hard CS problem;
        # in this case, has a "temperature" param that controls randomness;
        # the higher the value, the more randomness;
        # hihgher temperature => more creativity
    def send_message(self, user_message, temperature=0.0):
        """
        Sends a user message to the AI model and stores the pirate-themed response in history.
        
        >>> a = Chat()
        >>> cat = a.send_message('Say only the word HELLO and nothing else.')
        >>> 'HELLO' in cat
        True
        """
        
        self.messages.append({'role': 'user', 'content': user_message})
        while True:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=ALL_TOOL_SCHEMAS,
                temperature=temperature
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                self.messages.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = self.run_tool(tc.function.name, args)
                    self.messages.append({
                        'role': 'tool',
                        'tool_call_id': tc.id,
                        'content': result,
                    })
            else:
                content = msg.content or ''
                self.messages.append({'role': 'assistant', 'content': content})
                return content
def repl():
    '''
    Runs a terminal-based loop allowing users to interact with the pirate chat interface.
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
Hello, Monkey! 👋 How can I assist you today?
chat> Goodbye.
Goodbye! If you ever need anything else, just let me know. Have a great day!
<BLANKLINE> 
    '''
    import readline
    chat = Chat()
    try:
        while True:
            user_input = input('chat> ')
            response = chat.send_message(user_input, temperature=0.0)
            print(response)
    except KeyboardInterrupt:
        print()
    except EOFError:
        print()

if __name__ == '__main__':
    repl()


