"""Ling-3.0-flash gateway template tests: native roles, thinking cue, tools."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai_server import parse_ling30_tool_calls, render_chat_ling30


class Ling30ChatTemplate(unittest.TestCase):
    def test_roles_and_cue(self):
        prompt = render_chat_ling30(
            [{"role": "system", "content": "Be brief."},
             {"role": "user", "content": "Hi"}],
            enable_thinking=True)
        self.assertIn("<role>SYSTEM</role>\nBe brief.<|role_end|>", prompt)
        self.assertIn("<role>HUMAN</role>\nHi<|role_end|>", prompt)
        self.assertTrue(prompt.endswith("<role>ASSISTANT</role>\n<think>"))

    def test_thinking_off_closes(self):
        prompt = render_chat_ling30([{"role": "user", "content": "Hi"}],
                                    enable_thinking=False)
        self.assertTrue(prompt.endswith("<role>ASSISTANT</role>\n<think></think>"))

    def test_tool_declaration_and_observation(self):
        tools = [{"type": "function",
                  "function": {"name": "get_time",
                               "description": "now",
                               "parameters": {"type": "object",
                                              "properties": {"tz": {"type": "string"}}}}}]
        prompt = render_chat_ling30([{"role": "user", "content": "time?"}],
                                    tools=tools)
        self.assertIn("<tools>", prompt)
        self.assertIn("get_time", prompt)
        self.assertIn("<tool_call>", prompt)
        follow = render_chat_ling30(
            [{"role": "user", "content": "time?"},
             {"role": "tool", "content": "noon"}])
        self.assertIn("<role>OBSERVATION</role>\n<tool_response>noon</tool_response><|role_end|>",
                      follow)

    def test_parse_tool_calls(self):
        reply = ("thinking here <tool_call>get_time<arg_key>tz</arg_key>\n"
                 "<arg_value>UTC</arg_value></tool_call> done")
        content, calls = parse_ling30_tool_calls(
            reply, [{"type": "function",
                     "function": {"name": "get_time",
                                  "parameters": {"type": "object",
                                                 "properties": {"tz": {"type": "string"}}}}}])
        self.assertEqual(calls[0]["function"]["name"], "get_time")
        self.assertIn("UTC", calls[0]["function"]["arguments"])
        self.assertNotIn("<tool_call>", content)

    def test_continuation(self):
        prompt = render_chat_ling30(
            [{"role": "user", "content": "Hi"},
             {"role": "assistant", "content": "Hello"}],
            add_generation_prompt=False)
        self.assertIn("<role>ASSISTANT</role>\n<think></think>Hello", prompt)
        self.assertNotIn("<|role_end|><role>ASSISTANT</role>\n<think>",
                         prompt.split("<think></think>Hello")[-1])


if __name__ == "__main__":
    unittest.main()
