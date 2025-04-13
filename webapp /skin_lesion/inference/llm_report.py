import os
import re
import groq

client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

def clean_and_convert_markdown(text):
    """
    將 Markdown 格式的 **粗體** 轉換為 HTML 的 <b>粗體</b>，方便 PDF 呈現。
    """
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

def generate_case_report(label, confidence, location=None, age=None, duration=None, symptoms=None, size=None, gender=None):
    # 動態組合有效欄位
    details = []

    if gender:
        details.append(f"- 性別: {gender}")
    if age:
        details.append(f"- 年齡: {age}")
    if location:
        details.append(f"- 發病位置: {location}")
    if duration:
        details.append(f"- 發病時間長度: {duration}")
    if symptoms:
        details.append(f"- 症狀描述: {symptoms}")
    if size:
        details.append(f"- 病灶大小: {size}")

    details.append(f"- 疾病類型: {label}")
    details.append(f"- 模型信心值: {confidence:.2f}%")

    prompt = f"""
你是一位皮膚科醫師，請根據以下資訊撰寫一段中立、專業、條理清晰的繁體中文病歷描述，不要包含英文或「未填寫」的文字，也請避免多餘的標題或註解：

{chr(10).join(details)}

請以病歷書寫風格回覆，不要加註「以下是病歷描述」等前綴，直接書寫內容即可。
    """

    chat_completion = client.chat.completions.create(
        model="llama3-70b-8192",  # 可根據需要改其他 Groq 模型
        messages=[
            {"role": "system", "content": "你是一位皮膚科醫師，專業撰寫病歷。"},
            {"role": "user", "content": prompt}
        ]
    )

    raw_report = chat_completion.choices[0].message.content.strip()
    cleaned_report = clean_and_convert_markdown(raw_report)

    return cleaned_report