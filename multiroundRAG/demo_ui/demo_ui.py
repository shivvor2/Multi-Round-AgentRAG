import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from typing import Callable
from ..context_management.pool_of_queries import PoolOfQueries

# DISCLAIMER:
# THIS IS FOR TESTING ONLY, WRITTEN BY LLM (NOT ME)

class DemoChatUI:
    def __init__(self, response_inference_function: Callable, pool_of_queries: PoolOfQueries, inference_sysprompt: dict):
        self.response_inference_function = response_inference_function
        self.pool_of_queries = pool_of_queries
        self.inference_sysprompt = inference_sysprompt
        self.message_history = []

        self.output = widgets.Output()
        self.text_input = widgets.Text(placeholder='Type your message here...')
        self.send_button = widgets.Button(description='Send')
        self.role_checkbox = widgets.Checkbox(description='Show roles', value=False)
        self.reset_button = widgets.Button(description='Reset Chat')

        self.send_button.on_click(self.on_send)
        self.text_input.on_submit(self.on_send)
        self.role_checkbox.observe(self.update_chat_display, names='value')
        self.reset_button.on_click(self.reset_chat)

        input_box = widgets.HBox([self.text_input, self.send_button])
        input_box.layout.display = 'flex'
        self.text_input.layout.flex = '1'

        bottom_box = widgets.HBox([self.reset_button, self.role_checkbox])
        bottom_box.layout.display = 'flex'
        bottom_box.layout.justify_content = 'space-between'

        self.chat_box = widgets.VBox([self.output, input_box, bottom_box])
        self.main_output = widgets.Output()

        with self.main_output:
            display(self.chat_box)

        display(self.main_output)

    def on_send(self, _):
        user_message = self.text_input.value
        if user_message.strip():

            current_msg = self.make_user_message_dict(user_message)
            self.message_history.append(current_msg)
            self.pool_of_queries.update(self.message_history)
            self.text_input.value = ''

            current_context_msg = self.pool_of_queries.current_context_msg()
            new_msg = self.response_inference_function([self.inference_sysprompt] + [current_context_msg] + self.message_history)
            self.message_history.append(new_msg)

            self.update_chat_display()

    def add_message(self, role, content):
        self.message_history.append({"role": role, "content": content})
        self.update_chat_display()

    def update_chat_display(self, _=None):
        self.output.clear_output()
        with self.output:
            for message in self.message_history:
                role = message['role']
                content = message['content']

                if role == 'user':
                    align = 'right'
                    color = '#DCF8C6'
                elif role == 'assistant':
                    align = 'left'
                    color = '#E5E5EA'
                else:
                    align = 'left'
                    color = '#F3E5F5'

                role_display = f"<small>{role}: </small>" if self.role_checkbox.value else ""

                display(HTML(f"""
                    <div style="text-align: {align};">
                        <div style="display: inline-block; background-color: {color}; padding: 5px 10px; border-radius: 10px; max-width: 70%;">
                            {role_display}{content}
                        </div>
                    </div>
                """))

    def reset_chat(self, _):
        self.message_history = []
        self.text_input.value = ''
        self.role_checkbox.value = False
        self.pool_of_queries = type(self.pool_of_queries)()  # Create a new instance of the same type
        self.update_chat_display()

    @staticmethod
    def make_user_message_dict(new_message: str):
        return {
            "role": "user",
            "content": new_message
        }