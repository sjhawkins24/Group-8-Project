import seaborn as sns
import pandas as pd
<<<<<<< HEAD
=======


# update/add code below ...
#Exercise 1 fibonacci sequence 
def fibonacci(n):
    '''Recursive function to return the nth digit in the fibonacci sequence'''
    if n == 0 or n == 1:
        return(n)
    return(fib(n-1) + fib(n - 2))

#One question, what's the upshot of using a recursive function over a more
#readable loop like this: 
def fib(n):
    for i in range(n+1): 
      if i == 0: 
          old_n1 = 0
          old_n2 = 0

      elif i == 1 or i == 2: 
            old_n1 = 0
            old_n2 = 1

      #print(f" {old_n1} plus {old_n2}") 
      bin = old_n2
      old_n2 = old_n1 + old_n2
      #print(old_n2)
      old_n1 = bin
      #print(old_n2)
    return(old_n2)

#Exercise 2 binary conversion 
def to_binary(n):
    '''Recursive function to convert a number to binary'''
    if n == 0:
    #    print(0)
        return(0)
    if n == 1:
      #  print(1)
        return(1)
    return(str(to_binary(n // 2 )) + str(to_binary(n % 2)) )

#Exercise 3 
def task_1():
    '''Function that will return the names of the columns in descending order of number of
    missing values'''
    #Start by reading in the data 
    url = 'https://github.com/melaniewalsh/Intro-Cultural-Analytics/raw/master/book/data/bellevue_almshouse_modified.csv'
    df_bellevue = pd.read_csv(url)
    #Create a mask for the ? gender values
    filter_missing = df_bellevue['gender'] == '?'
    #Assign the ? to nan
    df_bellevue.loc[filter_missing, "gender"] = float('nan') 
    #Get the list of columns sorted by missing value
    missing_val = df_bellevue.isna().sum().sort_values()
    #Return the index to get just the column names
    return(missing_val.index)

def task_2(): 
    '''Function to return the number of entries by year'''
    #Start by reading in the data 
    url = 'https://github.com/melaniewalsh/Intro-Cultural-Analytics/raw/master/book/data/bellevue_almshouse_modified.csv'
    df_bellevue = pd.read_csv(url)
    #Create a new column for the year, based on the first four digits of the date
    df_bellevue["year"] = df_bellevue["date_in"].str[:4]
    #Grouping by year, and then counting the date in column
    grouped_df = df_bellevue.groupby("year")["date_in"].count().to_frame()
    return(grouped_df)

def task_3():
    '''Function to return series that gives the average age by gender'''
    #Start by reading in the data 
    url = 'https://github.com/melaniewalsh/Intro-Cultural-Analytics/raw/master/book/data/bellevue_almshouse_modified.csv'
    df_bellevue = pd.read_csv(url)
    #Group by gender and then get the mean of age
    return_df = df_bellevue.groupby("gender")["age"].mean()
    return(return_df)
>>>>>>> main
