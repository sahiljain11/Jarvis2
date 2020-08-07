import datetime

# sDate = "07/27/2012"
# dtDate = datetime.datetime.strptime(sDate, "%m/%d/%Y")
# dtDate = (str(dtDate))
# print(dtDate) #2012-07-27 00:00:00
# print(dtDate[0:4]) #2012



eventend = "01/10/2020 13:00:00"
dtDate = datetime.datetime.strptime(eventend, "%m/%d/%Y %H:%M:%S")
dtDate = str(dtDate)
print(dtDate[0:10]+"T"+dtDate[11:]+"-06:00") #2020-01-10 13:00:00
# print(datestringg)
print(dtDate[11:])
#needs to be in form'2020-09-10T09:00:00-06:00'