from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.index, name="index"),
    path('loan_approval/', views.loan_approval, name="loan_approval"),
    path('titanic_prediction/', views.titanic_prediction, name="titanic_prediction"),
    path('salary_prediction/', views.salary_prediction, name="salary_prediction"),
    path('diabetes_prediction/', views.diabetes_prediction, name="diabetes_prediction"),
    path('student_performance/', views.student_performance, name="student_performance"),
    path('stock_prediction/', views.stock_prediction, name="stock_prediction"),
    path('blog1/', views.blog1, name="blog1"),
    path('blog2/', views.blog2, name="blog2"),
    path('blog3/', views.blog3, name="blog3"),
    path('corona_prediction/', views.corona_prediction, name="corona_prediction"),
    path('blog4/', views.blog4, name="blog4"),    
]
