import os
import sys
import random
import gradio as gr


from pathlib import Path
from loguru import logger
from typing import List, Dict
from hyperpyyaml import load_hyperpyyaml

from code.src.database import DataTools
from code.src.LLM import LLM
from code.tools.base_tools import extract_content_after_marker

LOCAL_DIR: Path = Path(__file__).parent
config_file_path = LOCAL_DIR.joinpath('config/config.yaml')
with open(config_file_path) as fin:
    hparams = load_hyperpyyaml(fin, overrides={'BASEPATH':str(LOCAL_DIR)})


# 配置logger日志
logger.remove()
logger.add("mianbaobao.log")
log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level}] {file}:{line} - {message}"
logger.add(sys.stdout, format=log_format, level="DEBUG", 
           filter=lambda record: record["level"].no < 40)
logger.add(sys.stderr, format=log_format, level="ERROR", 
           filter=lambda record: record["level"].no >= 40)


data_tools = DataTools(hparams, logger)
llm_model = LLM(hparams, logger)

def create_db(subject_name: str) -> List[Dict]:
    """创建索引数据库,并返回一个随机问题

    Args:
        subject_name (str): 科目名称

    Returns:
        List[Dict]: 随机问题
    """
    history = [{'role': 'system', 'content': '你是一个有经验的面试官,可以根据问题,面试者回答以及正确答案进行打分和解析'}]
    data_tools.create_db(subject_name)
    question_list = data_tools.question_dict['questions']
    sub_question = random.choice(question_list)
    history.append({'role':'assistant', 
                    'content': f"请回答下面的问题:\n {sub_question}\n"})
    return history

def model_response(user_answer: str, history: List[Dict], selected_subject: str):
    """处理用户回答并生成反馈与新问题
    
    Args:
        user_answer: 用户的回答文本
        history: 对话历史
        selected_subject: 当前选择的面试科目
    """
    # 提取上一个问题
    previous_question = extract_content_after_marker(history[-1]['content'])

    # 如果用户想要下一题
    if '下一题' in user_answer and len(user_answer) <= 4:
        next_question = random.choice(data_tools.question_dict['questions'])
        assistant_message = f"请回答下面的问题:\n{next_question}"
        history.append({'role':'assistant', 'content': assistant_message})
        yield assistant_message
        return

    try:
        assistant_response = ""
        for response in llm_model.question_response(
            model_question=previous_question,
            answer=user_answer, 
            history=history, 
            subject_name=selected_subject
        ):
            assistant_response += response
            yield assistant_response

        next_question = random.choice(data_tools.question_dict['questions'])
        complete_message = f"{assistant_response}\n\n请回答下面的问题:\n{next_question}"

        history.append({'role':'assistant', 'content': complete_message})
        yield complete_message
    except Exception as e:
        error_message = f"生成回答时出错: {str(e)}"
        logger.exception(e)
        logger.error(error_message)
        yield error_message


with gr.Blocks() as demo:
    subject = gr.Radio(["rag", "llm", "code_ability", "ai_agents"])
    chat = gr.ChatInterface(model_response, 
                            additional_inputs=[subject], 
                            type="messages",
                            title="Mianshibaobao",
                            theme="ocean",)
    subject.change(create_db, subject, chat.chatbot_value)

demo.launch()


