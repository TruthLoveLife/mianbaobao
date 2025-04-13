# 面试宝宝

一个基于[LlamaIndex](https://docs.llamaindex.ai/en/stable/) + [gradio](https://www.gradio.app/) 构建的的面试助手小项目；

## 给你带来什么

- 构建自己的面试小助手，基于大模型和RAG对你的回答做出反馈和建议，成为你求职路上的好伙伴；
- 用来学习检索增强生成(RAG)技术，此项目可以用来初步练手和学习;
- 包含大量关于大模型、Agent、RAG相关的面试题；
- 持续更新中...

## 功能特点

- 面试问题生成
- 答案评估
- 实时反馈


## 用您的百炼API Key代替YOUR_DASHSCOPE_API_KEY
```bash
echo "export DASHSCOPE_API_KEY='YOUR_DASHSCOPE_API_KEY'" >> ~/.bashrc
source ~/.bashrc
```

[https://help.aliyun.com/zh/model-studio/developer-reference/configure-api-key-through-environment-variables?spm=a2c4g.11186623.0.0.60907980CmaHg1](https://help.aliyun.com/zh/model-studio/developer-reference/configure-api-key-through-environment-variables?spm=a2c4g.11186623.0.0.60907980CmaHg1)

### 手动安装

1. 克隆项目到本地

2. 创建虚拟环境：
   ```bash
   conda create -n mianbaobao python=3.10
   conda activate mianbaobao
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
4. 运行应用：
   ```bash
   python gr.py
   ```

## 项目结构

```
.
├── config/         # 配置文件
├── code/          # 源代码
├── data/          # 数据文件
├── main.py        # 命令行窗口测试文件
├── gr.py          # Gradio界面(主要)
├── requirements.txt # 依赖列表
```

## 使用说明

1. 启动应用后，在浏览器中访问 http://localhost:7860；
2. 在界面中输入面试问题或选择预设问题；
3. 系统会生成答案并提供评估；
4. 可以按照自己的需求，在data文件夹下，添加自己的准备的面试文档，同时按需更改配置文件中的 CoveringVectors 和 subjects；
