from django.shortcuts import render
from django.shortcuts import render
from django.http import JsonResponse
from django.http import Http404
import requests

def student_view(request):
    return render(request, "frontend/student.html")

def teacher_view(request):
    return render(request, "frontend/CourseList.html")

def login_view(request):
    return render(request, 'frontend/login.html')

def register_view(request):
    return render(request, 'frontend/register.html')

def knowledge_graph_view(request):
    """
    渲染前端的知识图谱页面（含 ECharts 图谱容器）
    """
    # 从URL获取参数，传给模板，前端再用JS拿参数调用后端接口
    course_name = request.GET.get('course_name', '')
    return render(request, 'frontend/knowledge_graph.html',{'course_name': course_name})

def course_page(request):
    return render(request, "frontend/Course.html")

def repair_view(request):
    return render(request, 'frontend/repair.html')
