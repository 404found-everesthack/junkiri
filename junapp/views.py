from django.shortcuts import render, redirect


# Create your views here.
from junapp.models import Runoff, Raindata


def checkifreached(request):
    return render(request,template_name='junapp/input.html')

def checkharvesting(request):
    runoff=Runoff.objects.filter(id=request.POST.get('type'))
    for i in runoff:
        coefficient=i.coefficient
    months=Raindata.objects.all()
    totalwater=0
    monthly=[]
    max=0
    for i in months:
        rainwater=coefficient*i.evaporation*i.rainfall*float(request.POST.get('area'))
        if (max<rainwater):
            max=rainwater
        monthly.append({'month':i.month,'rainfall':round(rainwater,2),'usage':request.POST.get('use')})
        # monthly[i.month]=round(rainwater,2)
        totalwater=totalwater+rainwater
    result={
        'totalwater':round(totalwater,2),
        'monthly':monthly,
        'max':max
    }

    print(monthly)
    return render(request,'junapp/result.html',{'result':result})
