string = 'helloworld'
res = ''.join(format(ord(char), '08b') for char in string)
print(res)
