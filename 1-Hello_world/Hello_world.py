# ====================================================
# File: Hello_world.py
# Author: G_T
# Version: 1.0
# Description: Demonstrates various ways to output "Hello world"
#              using functions, lambda expressions, classes, and string ops.
# ====================================================

def hi(): return "hello world"  # Return string
print(hi())  # Print function result

def hi1():
    global a  # Make 'a' global for use outside this function
    a = "!!!"
    return f"Hello world{a}"  # Return f-string including global 'a'
print(hi1())

lmb = lambda: f"Hi world{a}"  # Lambda returning f-string using global 'a'
print(lmb())

class Hello: 
    def helloMeth(self): return "Hello world - Class"  # Class method returning greeting
print(Hello().helloMeth())

class HelloWorld:
    def __str__(self): return "Hello world-Class Str"  # __str__ method returns greeting
print(HelloWorld())

class CallHell:
    def __call__(self): return "Hello world Class Call"  # Makes instance callable like a function
call = CallHell()
print(call())

class Hello3:
    def __init__(self, word="Hello", word1="World", ex=a):
        self.word, self.word1, self.ex = word, word1, ex  # Initialize with global 'a'
    def __str__(self): return f"{self.word} in my {self.word1}{self.ex}hehe{self.ex}"  # Formatted string
print(Hello3())

message = "hello world!!!"
print(r"HelloWorld")  # Print raw string literal
print(message[:11])  # Slice: first 11 characters
print(message.split())  # Split into list
print("%s %s" % ("Hello", "world"))  # Old-style formatting
print("{} {}".format("world", "Hello"))  # format() method
print(message.rstrip("!"))  # Remove trailing "!" characters
print(" ".join(message))  # Join characters with a space
print(message.replace("!!!", "???"))  # Replace substring
print(message.partition("l")[0])  # Partition and return first part
print("".join(list(message)[:6])[::-1])  # Join first 6 characters from list, reverse
print(" ".join(map(lambda x: x, ["Hello", "world"])))  # Map and join list elements

def Hi_gen():
    yield "olleH dlroW"  # Generator yielding a reversed greeting
print(next(Hi_gen()))

import sys
sys.stdout.write("hellO World\n")  # Write directly to stdout
