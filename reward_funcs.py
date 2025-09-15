import re

#second
# 提取 <answer> 中的内容
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# 从 gsm8k 原始答案里提取 "#### " 后的结果
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# 1) 正确性奖励：答案完全一致给 2 分
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    # completions 可能是 List[str] 或 List[List[{"content": ...}]]
    responses = [c if isinstance(c, str) else c[0].get("content", "") for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

# 2) 答案是整数奖励：给 0.5 分
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [c if isinstance(c, str) else c[0].get("content", "") for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted]

# 3) 严格格式奖励：检查严格换行和标签
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [c if isinstance(c, str) else c[0].get("content", "") for c in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

# 4) 宽松格式奖励：只要有标签即可
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c if isinstance(c, str) else c[0].get("content", "") for c in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

# 5) XML 标签完整度奖励
def count_xml(text: str) -> float:
    score = 0.0
    if text.count("<reasoning>\n") == 1: score += 0.125
    if text.count("\n</reasoning>\n") == 1: score += 0.125
    if text.count("\n<answer>\n") == 1: score += 0.125
    if text.count("\n</answer>") == 1: score += 0.125
    return score

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    responses = [c if isinstance(c, str) else c[0].get("content", "") for c in completions]
    return [count_xml(r) for r in responses]
