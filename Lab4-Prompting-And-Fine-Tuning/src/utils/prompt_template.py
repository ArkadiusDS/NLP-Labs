
class PromptTemplate:
    def __init__(self, system_prompt, user_prompt_template):
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def format_user_prompt(self, text):
        return self.user_prompt_template.format(text=text)