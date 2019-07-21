"""junkiri URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from junapp import views
# from junapp.views import checkifreached, checkharvesting,logi,inputTest
from junapp.views import checkharvesting, rainwaterharvest

urlpatterns = [
    path('rainwater-harvest',rainwaterharvest,name='rainwater-harvest'),
    path('check-harvesting',checkharvesting, name='check-harvesting'),
    path('logistic',logi, name='logistic'),
    path('inputTest', inputTest, name='inputTest')

]