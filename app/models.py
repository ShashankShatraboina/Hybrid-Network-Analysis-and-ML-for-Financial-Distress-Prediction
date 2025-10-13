from django.db import models

# Create your models here.
class User(models.Model):
	id = models.AutoField(primary_key=True)
	firstname= models.CharField(max_length=250)
	password= models.CharField(max_length=250)
	email= models.EmailField(unique=True)
	age= models.IntegerField()
	gender= models.CharField(max_length=250)
	phone= models.CharField(max_length=250)
	address= models.TextField()
class User1(models.Model):
	id = models.AutoField(primary_key=True)
	firstname= models.CharField(max_length=250)
	password= models.CharField(max_length=250)
	email= models.EmailField(unique=True)
	age= models.IntegerField()
	gender= models.CharField(max_length=250)
	phone= models.CharField(max_length=250)
	address= models.TextField()	   