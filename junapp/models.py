from django.db import models

# Create your models here.
class Runoff(models.Model):
    description=models.CharField(max_length=250)
    coefficient = models.FloatField()
    def __str__(self):
        return self.description

class Raindata(models.Model):
    month=models.CharField(max_length=250)
    rainfall=models.FloatField()
    evaporation=models.FloatField()
    days=models.IntegerField()
    def __str__(self):
        return self.month

class Testinput(models.Model):
    testinput = models.FileField(upload_to='junapp/static/junapp/data', default='', blank=True)

class hospital(models.Model):
    ward_no = models.IntegerField()
    address = models.CharField(max_length=200)
    type = models.IntegerField()
    passed = models.IntegerField()
    lat = models.FloatField()
    log = models.FloatField()

