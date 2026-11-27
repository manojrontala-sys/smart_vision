from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User, auth
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from .models import UserProfile
import re

# Create your views here.


def is_username_valid(username):
    username_pattern = re.compile(r'^[a-zA-Z0-9._-]+$')
    if username_pattern.match(username):
        return True
    return False

def is_email_valid(email):
    email_pattern = re.compile(r'^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}$')
    if email_pattern.match(email):
        return True
    return False

def signup(request):
    if request.method == 'POST':
        try:
            username = request.POST['username']
            email = request.POST['email']
            password = request.POST['password']
            confirm_password = request.POST['confirm_password']

            if not is_username_valid(username):
                messages.info(request, 'Invalid username!')
                return redirect('signup')

            if not is_email_valid(email):
                messages.info(request, 'Invalid email address!')
                return redirect('signup')

            if password == confirm_password:
                if User.objects.filter(email=email).exists():
                    messages.info(request, 'Email Taken')
                    return redirect('signup')
                elif User.objects.filter(username=username).exists():
                    messages.info(request, 'Username Taken')
                    return redirect('signup')
                else:
                    user = User.objects.create_user(username=username, email=email, password=password)
                    user_login = auth.authenticate(request, username=username, password=password)
                    if user_login is not None:
                        auth.login(request, user_login)

                    UserProfile.objects.create(user=user)

                    return redirect('dashboard')
            else:
                messages.info(request, 'Password Not Matching')
                return redirect('signup')

        except Exception as e:
            messages.info(request, 'Something went wrong. Please try again')
            return redirect('signup')
    else:
        return render(request, 'signup.html')


def signin(request):
    if request.method == 'POST':
        try:
            username = request.POST['username']
            password = request.POST['password']

            user = auth.authenticate(request, username=username, password=password)

            if user is not None:
                auth.login(request, user)
                return redirect('dashboard')
            else:
                messages.info(request, 'Credentials Invalid')
                return redirect('signin')

        except Exception as e:
            messages.info(request, 'Something went wrong. Please try again')
            return redirect('signin')

    else:
        return render(request, 'signin.html')


@login_required(login_url='signin')
def logout(request):
    auth.logout(request)
    return redirect('signin')