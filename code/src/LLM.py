import os


from pathlib import Path
from typing import List, Dict

from openai import OpenAI
from loguru._logger import Logger
from llama_index.core import StorageContext,load_index_from_storage,Settings
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank

from .database import DataTools

class LLM:
    def __init__(self, config: Dict, logger: Logger) -> None:
        self.config: Dict = config
        self.DataPath: Path = Path(self.config['Data']['DataPath'])
        self.logger = logger
        self.api_key = self.config['LLM']['api_key']
        self.model_name = self.config['LLM']['model_name']
        self.max_tokens = self.config['LLM']['max_tokens']
        self.temperature = self.config['LLM']['temperature']
        self.dashscope_url = self.config['LLM']['dashscope_url']
        self.history_round = self.config['LLM']['history_rounds']
        self.recalled_chunks = self.config['RAG']['recalled_chunks']
        self.similarity_threshold = self.config['RAG']['similarity_threshold']

    def model_response(self, query: str,                        
                       history: List, 
                       subject_name: str):
        subject_index = self.DataPath.joinpath(f"{subject_name}/db_save")
        if len(list(subject_index.glob('*.json'))) == 0:
            self.logger.info(f"{subject_index} has no index.json, now creating...")
            datatools = DataTools(self.config, self.logger)
            subject_index = datatools.create_db(subject_name)
        try:
            dashscope_rerank = DashScopeRerank(top_n=self.recalled_chunks, 
                                               return_documents=True)
            index_db = StorageContext.from_defaults(persist_dir=subject_index)
            index = load_index_from_storage(index_db)
            retriever_engine = index.as_retriever(
                similarity_top_k=20,
            )
            retrieve_chunk = retriever_engine.retrieve(query)
            try:
                results = dashscope_rerank.postprocess_nodes(retrieve_chunk, 
                                                             query_str=query)
            except Exception as e:
                self.logger.error(f"dashscope rerank error: {e}")
                results = retrieve_chunk[:self.recalled_chunks]
            search_text = ""
            for i in range(len(results)):
                if results[i].score >= self.similarity_threshold:
                    search_text = search_text + f"## {i+1}:\n {results[i].text}\n"
            prompt_template = f"按照大模型自身的知识并参考以下内容：{search_text}，以合适的语气回答用户的问题：{query}。不知道的地方可以回答不知道。一定不能乱说"
        except Exception as e:
            self.logger.error(f"load index from storage error: {e}")
            prompt_template = query

        # client = OpenAI(
        #         api_key=self.api_key,
        #         base_url=self.dashscope_url
        #     )
        client = OpenAI(
                base_url=self.dashscope_url
            )

        history_round = min(len(history), self.history_round)*2
        history.append({'role': 'user', 'content': prompt_template})
        completion = client.chat.completions.create(
            model=self.model_name ,
            messages=history[-history_round:],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
            )
        for chunk in completion:
            yield chunk.choices[0].delta.content

    def question_response(self, 
                        model_question: str,
                        answer: str,                        
                       history: List, 
                       subject_name: str):

        subject_index = self.DataPath.joinpath(f"{subject_name}/db_save")
        if len(list(subject_index.glob('*.json'))) == 0:
            self.logger.info(f"{subject_index} has no index.json, now creating...")
            datatools = DataTools(self.config, self.logger)
            subject_index = datatools.create_db(subject_name)
        try:
            dashscope_rerank = DashScopeRerank(top_n=self.recalled_chunks, 
                                               return_documents=True)
            index_db = StorageContext.from_defaults(persist_dir=subject_index)
            index = load_index_from_storage(index_db)
            retriever_engine = index.as_retriever(
                similarity_top_k=20,
            )

            retrieve_chunk = retriever_engine.retrieve(model_question)
             
            results = dashscope_rerank.postprocess_nodes(retrieve_chunk, 
                                                        query_str=model_question)

            search_text = ""
            for i in range(len(results)):
                if results[i].score >= self.similarity_threshold:
                    search_text = search_text + f"## {i+1}:\n {results[i].text}\n"

            prompt_template = f"""你是一个面试者,提问的问题是#{model_question}#\n, 正确答案参考#{search_text}#\n, 面试者的回答是:#{answer}#\n, 
                                基于你自身的能力和参考答案, 1,对面试者的回答首先打分(0-10分, 分母满分,分子得分); 2, 对面试者的回答进行解析并提出建议; 
                                3,给出正确答案。#注意#：不清楚的地方不用回答，一定不要乱说，采用markdown格式回答"""

        except Exception as e:
            self.logger.error(f"load index from storage error: {e}")
            prompt_template = f"""你是一个面试者,提问的问题是#{model_question}#\n, 面试者的回答是:#{answer}#\n, 
                    基于你自身的能力和参考答案, 1,对面试者的回答首先打分(0-10分, 分母满分,分子得分); 2, 对面试者的回答进行解析并提出建议;3,给出正确答案。
                    #注意#：不清楚的地方不用回答，一定不要乱说，采用markdown格式回答"""
        client = OpenAI(
                api_key=self.api_key,
                base_url=self.dashscope_url
            )
        history_round = min(len(history), self.history_round)*2
        history.append({'role': 'user', 'content': prompt_template})
        completion = client.chat.completions.create(
            model=self.model_name ,
            messages=history[-history_round:],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
            )
        for chunk in completion:
            yield chunk.choices[0].delta.content

        

            
        
        
        
