from django.shortcuts import render
from inference.predictor import run_prediction
from inference.llm_report import generate_case_report
from django.http import HttpResponse
from django.template.loader import get_template
from weasyprint import HTML
from django.utils.safestring import mark_safe
from django.conf import settings

import os
from PIL import Image
import numpy as np

def index(request):
    return render(request, 'index.html')

def predict_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']

        # ✅ 不儲存圖片，直接用 PIL 開啟記憶體中的圖片
        image = Image.open(image_file).convert('RGB')
        image = image.resize((224, 224))  # ⚠️ 根據你的模型調整尺寸
        image_array = np.array(image, dtype=np.float32) / 255.0

        # ✅ 推論：請確認 run_prediction 支援 numpy array
        label, confidence = run_prediction(image_array)

        # ✅ 表單欄位
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        location = request.POST.get('location')
        duration = request.POST.get('duration')
        symptoms = request.POST.get('symptoms')
        size = request.POST.get('size')

        # ✅ 呼叫 GPT LLM 生成病歷描述
        report = generate_case_report(
            label, confidence * 100,
            location=location,
            age=age,
            gender=gender,
            duration=duration,
            symptoms=symptoms,
            size=size
        )

        return render(request, 'result.html', {
            'label': label,
            'confidence': f"{confidence * 100:.2f}%",
            'report': mark_safe(report),
            'age': age,
            'gender': gender,
            'location': location,
            'duration': duration,
            'symptoms': symptoms,
            'size': size
        })

    return render(request, 'index.html', {'error': '請上傳圖片'})


def download_pdf(request):
    # ✅ 取得 GET 傳來的參數
    label = request.GET.get('label')
    confidence = request.GET.get('confidence')
    age = request.GET.get('age')
    gender = request.GET.get('gender')
    location = request.GET.get('location')
    duration = request.GET.get('duration')
    symptoms = request.GET.get('symptoms')
    size = request.GET.get('size')
    report = request.GET.get('report')

    # ✅ 準備 context
    context = {
        'label': label,
        'confidence': confidence,
        'age': age or '（未填寫）',
        'gender': gender or '（未填寫）',
        'location': location or '（未填寫）',
        'duration': duration or '（未填寫）',
        'symptoms': symptoms or '（未填寫）',
        'size': size or '（未填寫）',
        'report': mark_safe(report)
    }

    # ✅ 渲染 PDF
    html_template = get_template('pdf_template.html').render(context)
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="skin_diagnosis_report.pdf"'
    HTML(string=html_template, base_url=request.build_absolute_uri()).write_pdf(response)
    return response