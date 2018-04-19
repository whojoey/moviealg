#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:57:58 2018

@author: jamahlreynolds
""" 


import csv

def createProduction(myTitle, myDate, myGenre, myBudget, myRevenue): 

    with open('future_movies.csv', 'a', newline='') as f:
    
        thewriter = csv.writer(f)
    #fieldnames = ["original_title", "release_date", "budget", "revenue", "genres"] #mine 
  #  thewriter.writerow(['title', 'date', 'budget', 'revenue', 'genre'])
        thewriter.writerow([myTitle, myDate, myGenre, myBudget, myRevenue])
        
        
        
print ("Create New Movie? Y/N or Yes/No")
response = input()

if response.lower() in ['y', 'yes']:


    print ("enter values for each")
    myTitle = input ("\nTitle: ")
    myDate = input("\nDate: ")
    myGenre = input("\nGenre: ")
    myBudget = input("\nBudget: ")
    myRevenue = input("\nRevenue: ")
    
    createProduction(myTitle, myDate, myGenre, myBudget, myRevenue)



        
    
    # SHOULD WE USE A DICTIONARY FORMAT??
    #https://www.youtube.com/watch?v=s1XiCh-mGCA
    



with open('future_movies.csv') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        print (row)