import os
import re
import sys
import json

from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from loguru._logger import Logger

from llama_index.core import VectorStoreIndex,Settings,SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)

from code.tools.base_tools import *

EMBED_MODEL = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT
)
Settings.embed_model = EMBED_MODEL

class DataTools:
    def __init__(self, config: Dict, logger: Logger):
        self.config: Dict = config
        self.DataPath: Path = Path(self.config['Data']['DataPath'])
        self.logger = logger
        self.question_dict = {'questions': []}  # 问题列表
        self.api_key = self.config['LLM']['api_key']
        self.dashscope_url = self.config['LLM']['dashscope_url']

    def create_db(self, subject_name: str) -> Path:
        """创建索引数据库

        Args:
            subject_name (str): 科目名称

        Returns:
            Path: 索引数据库地址
        """
        original_subject_files = self.DataPath.joinpath(f"{subject_name}/original_file")
        save_db_path = self.DataPath.joinpath(f"{subject_name}/db_save")  # Save vector database address
        if not save_db_path.joinpath('default__vector_store.json').exists():
            save_db_path.mkdir(parents=True, exist_ok=True)
        else:
            if not self.config['Data']['CoveringVectors']:
                self.logger.info(f"{save_db_path} already exists, if you want to overwrite, please set CoveringVectors to True")
                with open(save_db_path.joinpath('questions.json'), 'r') as f:
                    self.question_dict = json.load(f)
                return save_db_path

        if not original_subject_files.exists():
            self.logger.error(f"{original_subject_files} not exists")
            raise FileNotFoundError(f"{original_subject_files} not exists")


        documents = []
        # for original_file in original_subject_files.iterdir():
        #     print(f"Loading {original_file}")
        documents.extend(SimpleDirectoryReader(input_dir=original_subject_files).load_data())

        self.extract_questions(documents, save_db_path, subject_name) # 提取问题
        self.logger.info(f"Extracted questions: {self.question_dict['questions']}")

        index = VectorStoreIndex.from_documents(
            documents
        )
        self.logger.info("Saving database ...")
        index.storage_context.persist(save_db_path)
        self.logger.info(f"Database {save_db_path} created successfully")
        return save_db_path

    def extract_questions(self, documents: List[Document], 
                          db_save_path: Path, 
                          subject_name: str):
        """从文档中提取问题
        Args:
            documents (List[Document]): llamaindex 读取的Document对象列表
        """
        # 语意分割
        splitter = SemanticSplitterNodeParser(
            buffer_size=2,  # 避免内容重叠
            breakpoint_percentile_threshold=95,
            embed_model=EMBED_MODEL
        )

        # nodes = splitter.get_nodes_from_documents(documents)
        nodes = []
        # import pdb; pdb.set_trace();
        
        for doc in documents:
            nodes_per = splitter.get_nodes_from_documents([doc])
            nodes.extend(nodes_per)

        llm = OpenAI(
            api_key=self.api_key,
            base_url=self.dashscope_url
        )
        # llm = OpenAI(
        #     base_url=self.dashscope_url
        # )
        clean_prompt = """
                    请从基于文本:#文本:{text}#，精确提取面试的问题，问题不能脱离面试的主题 #主题:{subject}#。不能有上文约束，如不能出现类似"在上个示例中，从上面的内容中，在上述代码"等类似的内容。
                    只需返回问题本身，不要添加任何解释。如果遇到描述性语句则转化为面试问题形式。不符合提取问题的文本返回空字符串。语言采用中文。
                    """

        for node in nodes:
            response = llm.chat.completions.create(
                model='qwen-max',
                messages=[
                    {"role": "system", "content": "你是一个专业的面试官，擅长从文本中提取面试问题。"},
                    {"role": "user", "content": clean_prompt.format(text=node.text, subject=subject_name)}
                ]
            )
            print("模型输出是: ")
            print(response)
            question = str(response.choices[0].message.content).strip()
            questions = re.findall(r'\d+\.\s+(.*?)(?=\s*\d+\.|$)', question, flags=re.DOTALL)
            for clean_question in questions:
                if '？' in clean_question or '?' in clean_question:
                    self.question_dict['questions'].append(clean_question)

        questions_save = db_save_path.joinpath('questions.json')  # 保存问题地址
        with open(questions_save, 'w') as f:
            f.write(json.dumps(self.question_dict, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    rag_question_dir = "/disk/aig/yangjianlei/makelm/sorfware/mianshi_baobao/mianbaobao/data/RAG/original_file" 
    # split_question(rag_question_dir)
