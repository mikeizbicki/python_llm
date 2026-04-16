"""Tests for python_llm/chat.py"""
import ast
import builtins
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open



# Helpers to build fake Groq responses


def _make_text_response(content):
    """Return a fake Groq response with a plain text message (no tool calls)."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    response = MagicMock()
    response.choices[0].message = msg
    return response


def _make_tool_response(tool_name, tool_args, tool_call_id="tc1"):
    """Return a fake Groq response that requests one tool call."""
    tc = MagicMock()
    tc.id = tool_call_id
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(tool_args)

    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]
    response = MagicMock()
    response.choices[0].message = msg
    return response



# Fixtures


@pytest.fixture
def mock_groq():
    """Patch Groq so no real network calls are made."""
    with patch('python_llm.chat.Groq') as MockGroq:
        yield MockGroq


@pytest.fixture
def chat(mock_groq):
    """A Chat instance backed by a mocked Groq client."""
    from python_llm.chat import Chat
    return Chat()



# is_path_safe


class TestIsPathSafe:
    def test_simple_filename(self):
        from python_llm.chat import is_path_safe
        assert is_path_safe('README.md') is True

    def test_current_dir(self):
        from python_llm.chat import is_path_safe
        assert is_path_safe('.') is True

    def test_empty_string(self):
        from python_llm.chat import is_path_safe
        assert is_path_safe('') is True

    def test_nested_relative_path(self):
        from python_llm.chat import is_path_safe
        assert is_path_safe('python_llm/chat.py') is True

    def test_absolute_path_rejected(self):
        from python_llm.chat import is_path_safe
        assert is_path_safe('/etc/passwd') is False

    def test_parent_traversal_rejected(self):
        from python_llm.chat import is_path_safe
        assert is_path_safe('../secret.txt') is False

    def test_embedded_traversal_rejected(self):
        from python_llm.chat import is_path_safe
        assert is_path_safe('some/../file.txt') is False



# _eval_node


class TestEvalNode:
    def _parse(self, expr):
        return ast.parse(expr, mode='eval').body

    def test_addition(self):
        from python_llm.chat import _eval_node
        assert _eval_node(self._parse('2 + 2')) == 4

    def test_multiplication(self):
        from python_llm.chat import _eval_node
        assert _eval_node(self._parse('3 * 7')) == 21

    def test_subtraction(self):
        from python_llm.chat import _eval_node
        assert _eval_node(self._parse('10 - 4')) == 6

    def test_division(self):
        from python_llm.chat import _eval_node
        assert _eval_node(self._parse('10 / 4')) == 2.5

    def test_power(self):
        from python_llm.chat import _eval_node
        assert _eval_node(self._parse('2 ** 8')) == 256

    def test_unary_negation(self):
        from python_llm.chat import _eval_node
        assert _eval_node(self._parse('-5')) == -5

    def test_unary_plus(self):
        from python_llm.chat import _eval_node
        assert _eval_node(self._parse('+3')) == 3

    def test_modulo(self):
        from python_llm.chat import _eval_node
        assert _eval_node(self._parse('10 % 3')) == 1

    def test_floor_division(self):
        from python_llm.chat import _eval_node
        assert _eval_node(self._parse('10 // 3')) == 3

    def test_string_constant_raises(self):
        from python_llm.chat import _eval_node
        with pytest.raises(ValueError):
            _eval_node(self._parse('"hello"'))

    def test_name_node_raises(self):
        from python_llm.chat import _eval_node
        node = ast.parse('x', mode='eval').body  # ast.Name, not allowed
        with pytest.raises(ValueError):
            _eval_node(node)



# Chat.calculate


class TestCalculate:
    def test_integer_result(self, chat):
        assert chat.calculate('2 + 2') == '4'

    def test_float_division_keeps_dot_zero(self, chat):
        assert chat.calculate('100 / 4') == '25.0'

    def test_float_non_division_strips_dot_zero(self, chat):
        assert chat.calculate('5 * 5.0') == '25'

    def test_repeating_float(self, chat):
        assert chat.calculate('10 / 3') == '3.3333333333333335'

    def test_zero_division(self, chat):
        assert chat.calculate('1 / 0') == 'Error: division by zero'

    def test_syntax_error(self, chat):
        assert chat.calculate('1 + (2 *') == 'Error: invalid expression'

    def test_unsafe_expression(self, chat):
        assert chat.calculate('__import__("os")') == 'Error: invalid expression'

    def test_none_type_error(self, chat):
        assert chat.calculate('None + 1') == 'Error: invalid expression'

    def test_floor_division_result(self, chat):
        assert chat.calculate('10 // 3') == '3'

    def test_negative_number(self, chat):
        assert chat.calculate('-5 + 10') == '5'


# Chat.ls


class TestLs:
    def test_unsafe_path_rejected(self, chat):
        assert chat.ls('/etc') == 'Error: unsafe path'

    def test_traversal_rejected(self, chat):
        assert chat.ls('../other') == 'Error: unsafe path'

    def test_nonexistent_folder_returns_empty(self, chat):
        assert chat.ls('nonexistent_folder_xyz') == ''

    def test_valid_folder_returns_sorted_names(self, chat, tmp_path):
        (tmp_path / 'b.txt').write_text('b')
        (tmp_path / 'a.txt').write_text('a')
        result = chat.ls(str(tmp_path))
        lines = result.split('\n')
        assert lines == ['a.txt', 'b.txt']



# Chat.cat


class TestCat:
    def test_unsafe_path_rejected(self, chat):
        assert chat.cat('/etc/passwd') == 'Error: unsafe path'

    def test_traversal_rejected(self, chat):
        assert chat.cat('../secret.txt') == 'Error: unsafe path'

    def test_file_not_found(self, chat):
        assert chat.cat('nonexistent_file_xyz.txt') == 'Error: file not found'

    def test_reads_file(self, chat, tmp_path):
        f = tmp_path / 'hello.txt'
        f.write_text('hello world')
        # cat expects a relative-style safe path; use a real temp file
        result = chat.cat(str(f))
        # tmp_path is absolute, so is_path_safe will reject it —
        # test the underlying open logic via mock instead
        assert result == 'Error: unsafe path'  # absolute tmp path blocked

    def test_reads_relative_file(self, chat):
        """Mock open to test the happy path without touching the filesystem."""
        with patch('builtins.open', mock_open(read_data='file contents')):
            result = chat.cat('somefile.txt')
        assert result == 'file contents'

    def test_unicode_decode_error_falls_back(self, chat):
        """First open raises UnicodeDecodeError; second succeeds with utf-16."""
        m = mock_open(read_data='utf16 contents')
        m.side_effect = [UnicodeDecodeError('utf-8', b'', 0, 1, 'reason'), m.return_value]
        with patch('builtins.open', m):
            result = chat.cat('somefile.txt')
        assert result == 'utf16 contents'



# Chat.grep


class TestGrep:
    def test_unsafe_path_rejected(self, chat):
        assert chat.grep('def ', '/etc') == 'Error: unsafe path'

    def test_traversal_rejected(self, chat):
        assert chat.grep('def ', '../other') == 'Error: unsafe path'

    def test_invalid_pattern(self, chat):
        result = chat.grep('[invalid', 'chat.py')
        assert result.startswith('Error: invalid pattern')

    def test_no_match_returns_empty(self, chat):
        assert chat.grep('zzz_no_match_xyz', '.') == ''

    def test_match_in_file(self, chat, tmp_path):
        f = tmp_path / 'sample.txt'
        f.write_text('hello world\nfoo bar\n')
        result = chat.grep('hello', str(f))
        assert 'hello world' in result

    def test_no_match_in_file(self, chat, tmp_path):
        f = tmp_path / 'sample.txt'
        f.write_text('hello world\n')
        result = chat.grep('zzz_no_match', str(f))
        assert result == ''



# Chat.send_message


class TestSendMessage:
    def test_plain_text_response(self, chat, mock_groq):
        chat.client.chat.completions.create.return_value = (
            _make_text_response('Hello!')
        )
        result = chat.send_message('Hi')
        assert result == 'Hello!'

    def test_message_appended_to_history(self, chat, mock_groq):
        chat.client.chat.completions.create.return_value = (
            _make_text_response('Hi there')
        )
        chat.send_message('Hey')
        roles = [m['role'] for m in chat.messages]
        assert 'user' in roles
        assert 'assistant' in roles

    def test_empty_content_returns_empty_string(self, chat, mock_groq):
        chat.client.chat.completions.create.return_value = (
            _make_text_response(None)
        )
        result = chat.send_message('Hello')
        assert result == ''

    def test_tool_call_then_text_response(self, chat, mock_groq):
        """Model first calls 'calculate', then returns a plain text answer."""
        tool_resp = _make_tool_response('calculate', {'expression': '2 + 2'})
        text_resp = _make_text_response('The answer is 4.')
        chat.client.chat.completions.create.side_effect = [tool_resp, text_resp]

        result = chat.send_message('What is 2 + 2?')
        assert result == 'The answer is 4.'

    def test_tool_result_appended_to_history(self, chat, mock_groq):
        tool_resp = _make_tool_response('calculate', {'expression': '1 + 1'})
        text_resp = _make_text_response('2')
        chat.client.chat.completions.create.side_effect = [tool_resp, text_resp]

        chat.send_message('1 + 1?')
        roles = [m['role'] if isinstance(m, dict) else 'assistant_msg'
                 for m in chat.messages]
        assert 'tool' in roles



# Chat.run_tool  (dispatch table)


class TestRunTool:
    def test_dispatches_calculate(self, chat):
        result = chat.run_tool('calculate', {'expression': '3 + 3'})
        assert result == '6'

    def test_dispatches_ls(self, chat):
        result = chat.run_tool('ls', {'folder': 'nonexistent_xyz'})
        assert result == ''

    def test_dispatches_cat(self, chat):
        result = chat.run_tool('cat', {'path': 'nonexistent_xyz.txt'})
        assert result == 'Error: file not found'

    def test_dispatches_grep(self, chat):
        result = chat.run_tool('grep', {'pattern': '[invalid', 'path': '.'})
        assert result.startswith('Error: invalid pattern')



# repl()


class TestRepl:
    def test_keyboard_interrupt_exits_cleanly(self, mock_groq, capsys):
        from python_llm.chat import repl

        inputs = iter([KeyboardInterrupt])

        def fake_input(prompt):
            val = next(inputs)
            if val is KeyboardInterrupt:
                raise KeyboardInterrupt
            return val

        with patch('builtins.input', side_effect=fake_input):
            with patch('python_llm.chat.Chat') as MockChat:
                repl()

        captured = capsys.readouterr()
        assert captured.out == '\n'

    def test_eof_exits_cleanly(self, mock_groq, capsys):
        from python_llm.chat import repl

        with patch('builtins.input', side_effect=EOFError):
            with patch('python_llm.chat.Chat'):
                repl()

        captured = capsys.readouterr()
        assert captured.out == '\n'

    def test_sends_message_and_prints(self, mock_groq, capsys):
        from python_llm.chat import repl

        call_count = 0

        def fake_input(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 'Hello'
            raise KeyboardInterrupt

        mock_chat_instance = MagicMock()
        mock_chat_instance.send_message.return_value = 'Hi there!'

        with patch('builtins.input', side_effect=fake_input):
            with patch('python_llm.chat.Chat', return_value=mock_chat_instance):
                repl()

        captured = capsys.readouterr()
        assert 'Hi there!' in captured.out
