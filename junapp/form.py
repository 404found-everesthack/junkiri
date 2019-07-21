from django import forms
from .models import Testinput

class InputForm(forms.ModelForm):
    class Meta:
        model = Testinput
        fields = ('testinput',)
