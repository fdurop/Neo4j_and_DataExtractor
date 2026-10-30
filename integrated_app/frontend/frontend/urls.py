from django.urls import path
from . import views
from django.shortcuts import render

urlpatterns = [
    path("", views.student_view, name="student_home"),
    path("teacher/", views.teacher_view, name="teacher_home"),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('knowledge-graph/', views.knowledge_graph_view, name='knowledge_graph'),
    path("course/", views.course_page, name="course"),
    path('repair/', views.repair_view, name='repair')
]