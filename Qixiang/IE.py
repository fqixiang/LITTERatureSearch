# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:26:18 2021

"""

# import packages
import PyPDF2 as PDF
import re
#import tabula
from tkinter import *
import easygui

import tkinter as tk


# open the pdf file
object = PDF.PdfFileReader("1-s2.0-S0025326X18304132-main.pdf")

# get number of pages
NumPages = object.getNumPages()

# define keyterms
String = "trawl"

# extract text and do the search
for i in range(0, NumPages):
    PageObj = object.getPage(i)
    print("this is page " + str(i)) 
    Text = PageObj.extractText() 
    
    # print(Text)
    ResSearch = re.search(String, Text)
    
    print(ResSearch)
    
# Not the best way to do this!


#read tables with camelot


tables = tabula.read_pdf("1-s2.0-S0025326X18304132-main.pdf", pages = "all", multiple_tables = True);

#tables




#Finde multiple words in the text at the same time
p = PDF.PdfFileReader("1-s2.0-S0025326X18304132-main.pdf")

# get number of pages
NumPages = p.getNumPages()

#define keyterms; David, Final, End, Score, Birthday, Hello Ben

kTerm = "trawl, km, measurement".replace(", ", "|")
#kTerm = "km, sample, latitude, measurement".replace(", ", "|")

Tables = [None] *NumPages   # Tables containing the words we want or the user spedifies
#extract text and do the search
for i in range(0, NumPages):
    PageObj = p.getPage(i)
    print("Looking through page " + str(i))
    Text = PageObj.extractText()
    Result = re.search(kTerm,Text)

    if Result:
         print(f"{kTerm} found")
         
         
         Tables[i]= tabula.read_pdf("1-s2.0-S0025326X18304132-main.pdf", pages =i ,  multiple_tables = True);
            
    else:
         print("0")
            
#Tables

###########MY APP ################
window = tk.Tk()
greeting = tk.Label(text="Hello, Tkinter")
greeting.pack()
