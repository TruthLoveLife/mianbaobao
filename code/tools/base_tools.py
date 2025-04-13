import re

def extract_dict_data(text):
    # 正则表达式匹配最外层的大括号内容
    pattern = r'\{[^{}]*\}'
    matches = re.findall(pattern, text)
    return matches

def extract_content_after_marker(text, marker="请回答下面的问题:"):
    # 使用正则表达式匹配marker后面的内容
    pattern = re.compile(re.escape(marker) + r'(.*?)$', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    else:
        return None