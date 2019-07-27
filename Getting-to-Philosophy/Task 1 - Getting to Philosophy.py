#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests

# URL for a random article
# url = requests.get('http://en.wikipedia.org/w/index.php?title=Special:Random')

# First URL
url = requests.get('https://en.wikipedia.org/wiki/Greece')


# Finds the first link/title (broken when it has a country)
def get_next_URL(url): 
    page = url.text
    # Find the first <p> tag for the main body
    mainP = page[page.find('<p>'):page.find('<p>')+500]
    # Find the first href for the link
    url = mainP[mainP.find('<a href="/wiki/')+15:mainP.find('"',mainP.find('<a href="/wiki/')+15)]
    return url

# Gets page name from the url
newPage = get_next_URL(url)

counter = 0 #Keeps track of the iterations

if newPage == 'Philosophy':
    print("The Random page chosen was the Philosophy page. Isn't the universe cool?")
else:
    print(newPage)
    while newPage !='Philosophy':
        # Creates the next link to go to based upon the first link
        newURL = 'http://en.wikipedia.org/w/index.php?title=' + newPage
        # Creates the next response
        new_response = requests.get(newURL)
        # Gets the next link by Calling the get_next_URL function        
        newPage = get_next_URL(new_response)

        print(newPage)
        counter +=1

print('\nIt took %d times to get to the Philosophy page on Wikipedia. Thanks WideBot for the puzzle!'% counter)


# In[ ]:
