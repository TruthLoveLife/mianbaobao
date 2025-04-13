import os
import sys
import random

from pathlib import Path
from loguru import logger
from hyperpyyaml import load_hyperpyyaml

from code.src.database import DataTools
from code.src.LLM import LLM

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

def main():
    data_tools = DataTools(hparams, logger)
    llm_model = LLM(hparams, logger)

    while True:
        subject_name = input("请输入科目名称: ")
        if subject_name not in hparams['subjects']:
            logger.error(f"目前只支持的科目有: {hparams['subjects']}， 请重新输入")
            continue
        break

    data_tools.create_db(subject_name)
    question_list = data_tools.question_dict['questions']

    history = [{'role': 'system', 'content': '你是一个有经验的面试官,可以根据问题,面试者回答以及正确答案进行打分和解析'}]

    while True:
        sub_question = random.choice(question_list)
        print(f"\n新问题: {sub_question}")
        # history.append({'role': 'assistant', 'content': sub_question})
        user_ansewer = input("\n请回答问题: ")
        assistant_str = ""
        for response in llm_model.question_response(model_question = sub_question,
                                                    answer=user_ansewer, 
                                                    history = history, 
                                                    subject_name = subject_name):
            print(response, end='')
            assistant_str += response
        history.append({'role': 'assistant', 'content': assistant_str})

if __name__ == '__main__':
    main()