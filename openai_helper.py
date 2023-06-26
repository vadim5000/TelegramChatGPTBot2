import datetime
import logging
import openai


class OpenAIHelper:
    """
    Допоміжний клас для роботи з ChatGPT
    """

    def __init__(self, config: dict):
        """
        Ініціалізуємо клас з переданими конфігураціями
        :param config: Словник з конфігураціями для GPT
        """
        openai.api_key = config['api_key']
        self.config = config
        self.conversations: dict[int: list] = {}  # {chat_id: history}
        self.last_updated: dict[int: datetime] = {}  # {chat_id: last_update_timestamp}

    def get_chat_response(self, chat_id: int, query: str) -> str:
        """
        Обробка відповіді від ChatGPT
        :param chat_id: Айді чату
        :param query: Запит, який ми відсилаємо до GPT
        :return: Відповідь від чату
        """
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)

            self.last_updated[chat_id] = datetime.datetime.now()

            # Перевіряємо розмір історії, щоб не використовувати надлишково наш токен
            if len(self.conversations[chat_id]) > self.config['max_history_size']:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    summary = self.__summarise(self.conversations[chat_id])
                    logging.debug(f'Summary: {summary}')
                    self.reset_chat_history(chat_id)
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            self.__add_to_history(chat_id, role="user", content=query)

            response = openai.ChatCompletion.create(
                model=self.config['model'],
                messages=self.conversations[chat_id],
                temperature=self.config['temperature'],
                n=self.config['n_choices'],
                max_tokens=self.config['max_tokens'],
                presence_penalty=self.config['presence_penalty'],
                frequency_penalty=self.config['frequency_penalty'],
            )

            if len(response.choices) > 0:
                answer = ''

                if len(response.choices) > 1 and self.config['n_choices'] > 1:
                    for index, choice in enumerate(response.choices):
                        if index == 0:
                            self.__add_to_history(chat_id, role="assistant", content=choice['message']['content'])
                        answer += f'{index+1}\u20e3\n'
                        answer += choice['message']['content']
                        answer += '\n\n'
                else:
                    answer = response.choices[0]['message']['content']
                    self.__add_to_history(chat_id, role="assistant", content=answer)

                if self.config['show_usage']:
                    answer += "\n\n---\n" \
                              f"💰 Tokens used: {str(response.usage['total_tokens'])}" \
                              f" ({str(response.usage['prompt_tokens'])} prompt," \
                              f" {str(response.usage['completion_tokens'])} completion)"

                return answer
            else:
                logging.error('No response from GPT-3')
                return "⚠️ _An error has occurred_ ⚠️\nPlease try again in a while."

        except openai.error.RateLimitError as e:
            logging.exception(e)
            return f"⚠️ _OpenAI Rate Limit exceeded_ ⚠️\n{str(e)}"

        except openai.error.InvalidRequestError as e:
            logging.exception(e)
            return f"⚠️ _OpenAI Invalid request_ ⚠️\n{str(e)}"

        except Exception as e:
            logging.exception(e)
            return f"⚠️ _An error has occurred_ ⚠️\n{str(e)}"

    def reset_chat_history(self, chat_id):
        """
        Resets the conversation history.
        """
        self.conversations[chat_id] = [{"role": "system", "content": self.config['assistant_prompt']}]

    def __max_age_reached(self, chat_id) -> bool:
        """
        Checks if the maximum conversation age has been reached.
        :param chat_id: The chat ID
        :return: A boolean indicating whether the maximum conversation age has been reached
        """
        if chat_id not in self.last_updated:
            return False
        last_updated = self.last_updated[chat_id]
        now = datetime.datetime.now()
        max_age_minutes = self.config['max_conversation_age_minutes']
        return last_updated < now - datetime.timedelta(minutes=max_age_minutes)

    def __add_to_history(self, chat_id, role, content):
        """
        Adds a message to the conversation history.
        :param chat_id: The chat ID
        :param role: The role of the message sender
        :param content: The message content
        """
        self.conversations[chat_id].append({"role": role, "content": content})

    def __summarise(self, conversation) -> str:
        """
        Summarises the conversation history.
        :param conversation: The conversation history
        :return: The summary
        """
        messages = [
            { "role": "assistant", "content": "Summarize this conversation in 700 characters or less" },
            { "role": "user", "content": str(conversation) }
        ]
        response = openai.ChatCompletion.create(
            model=self.config['model'],
            messages=messages,
            temperature=0.4
        )
        return response.choices[0]['message']['content']
