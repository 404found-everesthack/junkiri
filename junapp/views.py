from django.shortcuts import render, redirect


# Create your views here.
from junapp.models import Runoff, Raindata


def rainwaterharvest(request):
    return render(request,template_name='junapp/input.html')

def checkharvesting(request):
    runoff=Runoff.objects.filter(id=request.POST.get('type'))
    for i in runoff:
        coefficient=i.coefficient
    months=Raindata.objects.all()
    totalwater=0
    monthly=[]
    max=0
    maxexpense=0
    totalsave=0
    for i in months:
        # Q=kRA
        rainwater=coefficient*i.evaporation*i.rainfall*float(request.POST.get('area'))
        if (max<rainwater):
            max=rainwater
        usage=int(request.POST.get('use'))*i.days
        waterrequired=usage-rainwater
        # calculating average water bill after rain harvesting
        if (waterrequired<500):
            monthlyprice=300
        elif (waterrequired<1000):
            monthlyprice=300+(1*(waterrequired-500))
        else:
            monthlyprice=300+(1*(500)+2*(waterrequired-1000))

        if (maxexpense<monthlyprice):
            maxexpense=monthlyprice

        # calculating average water bill before rain harvesting
        if (usage < 500):
            requiredmonthlyprice = 300
        elif (usage < 1000):
            requiredmonthlyprice = 300 + (1 * (usage - 500))
        else:
            requiredmonthlyprice = 300 + (1 * (500) + 2 * (usage - 1000))

        # total money saved per month
        totalsave=totalsave+(requiredmonthlyprice-monthlyprice)

        if (maxexpense<usage):
            maxexpense=usage

        monthly.append({'month':i.month,'rainfall':round(rainwater,2),'usage':usage,'monthlyprice':monthlyprice, 'requiredmonthlyprice':requiredmonthlyprice})
        # monthly[i.month]=round(rainwater,2)
        totalwater=totalwater+rainwater

    result={
        'totalwater':round(totalwater,2),
        'monthly':monthly,
        'max':max,
        'maxexpense':maxexpense,
        'totalsave':round(totalsave,2)
    }

    print(monthly)
    return render(request,'junapp/result.html',{'result':result})
