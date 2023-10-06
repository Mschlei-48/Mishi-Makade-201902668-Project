from django.shortcuts import render
from django.http import JsonResponse
from .models import Features
from .serializers import FeaturesSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

@api_view(['POST'])
def predict_sneaker(request):
    with open('C:/Users/Student/Desktop/Honours/Second-Semester/SpecialTopics/Project/Django/App/MLModel/Model.pkl', 'rb') as model_file:
        model, vectorizer = pickle.load(model_file)
    #This is the data the user will enter. You use the request method to access the data entered
    data=request.data['features']
    data = vectorizer.transform(data)
    prediction=model.predict(data)
    def final_pred(prediction):
        if prediction==np.array([1]) or prediction==np.array([2]):
            prediction="Negative"
        elif prediction==np.array([3]):
            prediction="Neutral"
        else:
            prediction="Positive"
        return prediction
    prediction=final_pred(prediction)

    return Response({'prediction':prediction})











# Create your views here.
@api_view(['GET','POST'])
def Features_list(request):
    if request.method=='GET': #We are requesting to view/display the data
        features=Features.objects.all()
        serializer=FeaturesSerializer(features,many=True) 
        return JsonResponse(serializer.data,safe=False)
    if request.method=='POST': #We are requesting to post data/ put new data in the database
        serializer=FeaturesSerializer(data=request.data)#Here we are saying the data that has been enterd bythe user through a post request is the data that must be serialized
        if serializer.is_valid():                       #Note taht in order for you to enter the data for the put request you will need to use a UI. Since we do not have an app crated, we willl use a webiste called post-man to enter the data through a UI.
            serializer.save()
            return Response(serializer.data,status=status.HTTP_201_CREATED)                           #When you have a website you can use the POSt request to et data from the user and post it in the website or do whatever you want to do with the data.
#With the line  we are checking if the data we get through the UI meets the requirements of the serializer. If it does then we save the data.
#Then at the last code we are saying once the dta has been validated to be corerct by the serializer we should return as reposne that lets us know tath everything has been created successfully.