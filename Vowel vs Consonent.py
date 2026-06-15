sentence = input("Please enter sentence:")
vowels = "aeiouAEIOU"
vowel_count = 0
consonent_count = 0

for char in sentence:
    if char.isalpha():
        if char in vowels:
            vowel_count += 1
        else:
            consonent_count += 1

print("Vowel: " +  str(vowel_count))
print("Consonent: " +  str(consonent_count))





# Age = 30
# Name = "Abubakr Tufail"
# Address = "Lahore, Pakistan"
# print("Hi, " + Name + " your age is " + str(Age) + " as per documents. Your given address is " + Address)


# thisdict = {
#   "brand": "Ford",
#   "model": "Mustang",
#   "year": 1964
# }
# print(thisdict["brand"])


# thisdict = {
#   "brand": "Ford",
#   "model": "Mustang",
#   "year": 2020,
#   "year": 1964,
# }
# print(thisdict)

# if "ab1" in "abubakr":
#     print('Found')
# else:
#     print('Not found')


