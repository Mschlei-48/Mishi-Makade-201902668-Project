from rest_framework import serializers
from .models import Features

class FeaturesSerializer(serializers.ModelSerializer):
    class Meta:
        model=Features
        fields=['review_text']