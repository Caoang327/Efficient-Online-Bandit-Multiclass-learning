import os
os.system("python Banditron_Nosys_com.py 1>>log1.txt")
os.system("python Banditron_Nosys_find_gamma.py 1>>log1.txt")
os.system("python PWNeutron_Nosys_com.py 1>>log2.txt")
os.system("python PWNetron_Nosys_find_gamma.py 1>>log2.txt")
os.system("python SOBA_Nosys_com.py  1>>log3.txt")
os.system("python PWNetron_Ruster.py  1>>log3.txt")