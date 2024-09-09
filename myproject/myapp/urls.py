
from django.urls import path
from .views import index, upload_pdf, ask_question

urlpatterns = [
    path('', index, name='index'),
    path('upload/', upload_pdf, name='upload_pdf'),
    path('ask/', ask_question, name='ask_question'),
]
