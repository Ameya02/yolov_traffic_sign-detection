import tkinter as tk
import tkinter.messagebox as msb
from twilio.rest import Client
from dotenv import load_dotenv
import datetime
import requests
import json


import os
tk.Tk().withdraw()
load_dotenv()
import random
def random_location():

    # Define lists of possible street names, cities, and states
    streets = ['Gandhi Road', 'Park Street', 'MG Road', 'Jawaharlal Nehru Road', 'Anna Salai', 'Bandra Kurla Complex']
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata']
    states = ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'Telangana', 'West Bengal']

    # Generate a random street number
    street_number = random.randint(1, 999)

    # Choose a random street name, city, and state
    street_name = random.choice(streets)
    city = random.choice(cities)
    state = random.choice(states)

    # Generate a random pin code
    pin_code = random.randint(100000, 999999)

    # Print out the generated address
    return f'{street_number}, {street_name}\n{city}, {state} - {pin_code}'

def sms_alert(name,license_plate,license_no,traffic_violation):
    curr_time = datetime.datetime.now()
    msb.showwarning(
        "Traffic Violated",
        "Dear "+ name +",\n"+
        " License plate "+license_plate + "\n"+
        " License No "+ license_no + "\n"+
        " You have violated the following traffic rules: \n" + traffic_violation +
        " @ " + random_location()+ "AT "+ curr_time.strftime("%c")
    )

def sms_alert_twilio(name,license_plate,license_no,traffic_violation):
    curr_time = datetime.datetime.now()
    sms= "Dear "+ name +",\n" + " License plate "+license_plate + "\n"+ " License No "+ license_no + "\n"+ " You have violated the following traffic rules: \n" + traffic_violation + " @ " + random_location()+ "AT "+ curr_time.strftime("%c")
    account_sid = os.getenv("ACC_SID")
    auth_token = os.getenv("AUTH_TOKEN")
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=sms,
        from_=os.getenv("TWILIO_NUMBER"),
        to=os.getenv("MY_PHONE")
    )
    print(message.sid)

def sms_alert_f2sms(name,license_plate,license_no,traffic_violation,mno,ph1):
    url = "https://www.fast2sms.com/dev/bulkV2"
    curr_time = datetime.datetime.now()
    msg= "Dear "+ name +",\n" + " License plate "+license_plate + "\n"+ " License No "+ license_no + "\n"+ " You have violated the following traffic rules: \n" + traffic_violation + " @ " + random_location()+ "AT "+ curr_time.strftime("%c")

    querystring = {
        'authorization': os.getenv("AUTH_F2S"),
        "message": msg,
        "language": "english",
        "route": "v3",
        "numbers": str(mno)+","+str(ph1)}

    headers = {
    'Cache-Control': "no-cache"
    }
    try:
        response = requests.request("GET", url,
                                    headers=headers,
                                    params=querystring)

        print("SMS Successfully Sent")
    except:
        print("Oops! Something wrong")




