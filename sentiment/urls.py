from django.urls import path
from .views import SentimentAnalysis

urlpatterns = [
    path('analyze/', SentimentAnalysis.as_view(), name='analyze'),
]
   
