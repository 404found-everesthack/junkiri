from django.shortcuts import render, redirect


# Create your views here.
from junapp.models import Runoff, Raindata


def checkifreached(request):
    return render(request,template_name='junapp/input.html')

def checkharvesting(request):
    result=[]
    runoff=Runoff.objects.filter(id=request.POST.get('type'))
    for i in runoff:
        coefficient=i.coefficient
    months=Raindata.objects.all()
    totalwater=0
    for i in months:
        totalwater=totalwater+(coefficient*i.evaporation*i.rainfall*request.POST.get('area'))


    result.append(request.POST.get('area'))
    result.append(request.POST.get('use'))
    result.append(request.POST.get('type'))
    return render(request,'junapp/result.html',{'result':totalwater})
