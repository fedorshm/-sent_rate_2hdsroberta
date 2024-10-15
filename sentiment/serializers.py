from rest_framework import serializers

class TextInputSerializer(serializers.Serializer):
    text = serializers.CharField()
