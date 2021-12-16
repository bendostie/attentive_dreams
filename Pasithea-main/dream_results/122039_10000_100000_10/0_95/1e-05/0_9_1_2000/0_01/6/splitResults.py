import re

f1 = open('originalMols.csv', 'w')
f2 = open('dreamedMols.csv', 'w')
f3 = open('originalMols_prop.csv', 'w')
f4 = open('dreamedMols_prop.csv', 'w')



with open('original_to_dream_mol', 'r') as f:
    lines = f.readlines()
    for line in lines:
        sections = re.split('--> |,', line)
        print(sections)
        f1.writelines(sections[0] + '\n')
        f2.writelines(sections[1] + '\n')
        f3.writelines(sections[2] + '\n')
        f4.writelines(sections[3] + '\n')
        print(sections)
        


