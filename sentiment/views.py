from django.shortcuts import render

import torch
from transformers import RobertaTokenizer, RobertaModel
from torch import nn
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .serializers import TextInputSerializer

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_labels_task1, num_labels_task2):
        super(MultiTaskModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier1 = nn.Linear(self.roberta.config.hidden_size, num_labels_task1)
        self.classifier2 = nn.Linear(self.roberta.config.hidden_size, num_labels_task2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        return logits1, logits2

class SentimentAnalysis(APIView):
    parser_classes = [MultiPartParser]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_name = 'Gnider/distillroberta_2heads_sentimrate'
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = MultiTaskModel(model_name, num_labels_task1=2, num_labels_task2=8)
        self.model.eval()

    def predict_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        with torch.no_grad():
            logits1, logits2 = self.model(**inputs)

        sentiment_probs = torch.softmax(logits1, dim=1).squeeze().tolist()
        sentiment_labels = ['negative', 'positive']
        sentiment_scores = {label: score for label, score in zip(sentiment_labels, sentiment_probs)}

        rating_probs = torch.softmax(logits2, dim=1).squeeze().tolist()
        top_3_ratings = sorted(range(len(rating_probs)), key=lambda i: rating_probs[i], reverse=True)[:3]
        top_3_scores = {str(rating): rating_probs[rating] for rating in top_3_ratings}

        return sentiment_scores, top_3_scores

    def post(self, request):
        if 'file' in request.data:
            file = request.data['file']
            text = file.read().decode('utf-8')
        else:
            serializer = TextInputSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            text = serializer.validated_data['text']

        sentiment_scores, top_3_scores = self.predict_text(text)
        return Response({
            'sentiment': sentiment_scores,
            'top_3_ratings': top_3_scores
        })
   
