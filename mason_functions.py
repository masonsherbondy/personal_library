#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#1
#is_two defines a single parameter, x (whatever user input), and determines whether or not that input is
#two (it returns a Boolean value of True or False)

def is_two(x):
    #check to make sure the input is explicitly the integer two or the numerical string for two
    if x == 2 or x == '2': 
        #return bool True if it is two
        return True
    else:
        #return bool False if not
        return False

is_two('2')


# In[3]:


#2
#is_vowel defines a single parameter, x (assuming it is a string of one character), and determines whether the input 
#is a vowel or not (returns Boolean True or False)
def is_vowel(x):
    #check to see if the input (lowercased just in case input is uppercased) is in this complete string of vowels
    if x.lower() in 'aeiou':
        #return bool True if it's in the string
        return True
    else:
        #return bool False if not
        return False
   
is_vowel('A')


# In[ ]:


#3
#is_consonant defines a single parameter, x (string of one character), and returns a Boolean True or False
def is_consonant(x):
    #a letter is either a consonant or a vowel; check to see if the input is a vowel
    if is_vowel(x):
        #return bool False if it is a vowel
        return False
    else:
        #return bool True if it is not a vowel
        return True
    
is_consonant('B')


# In[ ]:


#4
#capitalize_consonant_words defines a single parameter, a string, and returns a string capitalized
#if it begins with a consonant
def capitalize_consonant_words(string):
    #check to see if the first letter of the string is a consonant
    if is_consonant(string[0]) == True:
        #return the string capitalized if the first letter checks out
        return string.capitalize()
    else:
        #return the string as it was input; does not capitlize string if first letter not a consonant
        return string
    
capitalize_consonant_words('success')


# In[ ]:


#5
#calculate_tip defines two parameters, a float and another float, and returns a float value
def calculate_tip(percentage, bill):
    #set identifier 'tip' to the value of the product for the percentage (float 1) and the bill (float 2) 
    tip = bill * percentage
    #we just want the tip (rounded to two decimal places)
    return round(tip, 2)

calculate_tip(.22, 18.24)


# In[ ]:


#6 
#apply_discount defines two parameters, two floats, and returns a float value
def apply_discount(price, disc):
    #return the discounted price (price minus discount) as a float, rounded to two decimals
    return round(price * (1 - disc), 2)

apply_discount(24.99, .8)


# In[4]:


#7
#my handle_commas defines a single parameter, a string, and returns a float value
def handle_commas(string):
    #assign a variable to an empty string to work with
    c_handled = ''
    #start a loop to check every character in string input
    for c in string:
        #if the character is a number, add to the variable (string)
        if c.isdigit():
            c_handled += c
        #if the character is a comma, add an underscore in its stead
        elif c == ',':
            c_handled += '_'
        #if the character is a decimal, add to the string
        elif c == '.':
            c_handled += '.'
    #return the string as a float
    return float(c_handled)

handle_commas('12,345,678.9')


# In[ ]:


#8
#get_letter_grade determines a single parameter, an integer, and returns a string value
def get_letter_grade(x):
    #check if the integer is above the value of 89, return string value 'A' if it is
    if x > 89:
        return 'A'
    #since we are checking if the input is greater than a certain value, and python reads top-to-bottom, we can 
    #just check if the input is above a value less than the previous if-clause value and return the 
    #appropriate/ corresponding string value
    elif x > 79:
        return 'B'
    elif x > 69:
        return 'C'
    elif x > 64:
        return 'D'
    else:
        return 'F'   
    
get_letter_grade(69)


# In[ ]:


#9
#remove_vowels determines a single parameter, a string, and returns a string
def remove_vowels(string):
    #assign a variable to an empty string
    v_removed = ''
    #start a loop to check if characters in input string are vowel-opposites
    for l in string:
        if is_consonant(l):
            #only add vowel-opposite characters to the string variable
            v_removed += l
    #return the string variable with characters that are not vowels
    return v_removed

remove_vowels('Nnnnooooooooooo!!!!')


# In[2]:


#10
#normalize_name defines a single parameter, a string, and returns a string value
def normalize_name(string):
    #python identifiers can not start with numbers. Loop through string until first character of string is not 
    #a number
    while string[0].isalpha() == False:
        string = string[1:]
    #assign a variable to: the input string stripped of all leading or trailing white space, as well as with all
    #lowercased characters
    f_name = string.strip().lower() 
    #assign a variable to an empty string
    e_name = ''
    #twice
    end_game = ''
    #start a loop to see if string characters are python identifer compliant
    for char in f_name:
        #include space so we don't have random underscores at the beginning or end if original string input includes
        #invalid characters for python-identifier compliance at the beginning or end of original input and spaces
        #inbetween those invalid characters and valid ones. check to see if characters are in approved list
        if char in 'abcdefghijklmnopqrstuvwxyz_0123456789 ':
            #add all the approved characters to the first empty string variable
            e_name += char
    #assign a variable to the result stripped of all the leftover leading or trailing whitespace
    g_name = e_name.strip()
    #start another for loop to get a cleaner string
    for char in g_name:
        #check to see if character is py-id compliant
        if char in 'abcdefghijklmnopqrstuvwxyz_0123456789':
            #add py-id compliant characters to second empty string variable
            end_game += char
        #check to see if character is a space
        if char == ' ':
            #add an underscore to the second string variable in a space's stead
            end_game += '_'
    #return the python-identifier-compliant second string variable value
    return end_game

if __name__ == '__main__':
    print(normalize_name('First name'))
    print(normalize_name('Name'))
    print(normalize_name('% Completed'))


# In[1]:


#11
#Example code
#cumulative_sum([1, 1, 1]) returns [1, 2, 3]
#cumulative_sum([1, 2, 3, 4]) returns [1, 3, 6, 10]

#interpretation
#if A = [1, 1, 1], then cumulative_sum(A) = [A[0], (A[0] + A[1]), (A[0] + A[1] + A[2])]
#cumulative_sum(A) = [sum(A[:1]), sum(A[:2]), sum(A[:3])]
#cumulative_sum(A) = [sum(A[:0 + 1]), sum(A[:1 + 1]), sum(A[:2 + 1])]
#range(len(A)) = range(0, 3) --> which is zero to two (0, 1, 2)--> which is the complete index for A

#cumulative_sum defines a single parameter, a list, and returns a list
def cumulative_sum(L):
    #return a list of numbers that are each the sum of the number in the input list plus all of the numbers prior to
    #this number in the index. use python's zero-indexing to format the individual sums for the return list. 
    #use a for loop to run through the index of the input list and generate sums for each item on the return list.
    return [sum(L[:n + 1]) for n in range(len(L))]

c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cumulative_sum(c)


# In[2]:


#Mason's personal functions


# In[3]:


#pull a number from a string with one number 
def pull_a_number_from_a_string_with_one_number(string):
    number = ''
    for char in string:
        if char.isdigit():
            number += char
    return int(number)
pull_a_number_from_a_string_with_one_number('Yo! You have 38 unread messages!')
#good to alias it as 'pn'
#can copy and paste below after '#' marker:
#from mason_functions import pull_a_number_from_a_string_with_one_number as pn


# In[ ]:




