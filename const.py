# -*- coding: utf-8 -*-

'''
 $ cat input/structures.csv | cut -f3 -d, |sort | uniq -c
 831726 C
   2996 F
1208387 H
 132361 N
 183187 O
'''

# https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=html&isotype=some
atomic_weight = {
    'H': 1.00782503223,
    'C': 12.0000000,
    'N': 14.00307400443,
    'O': 15.99491461957,
    'F': 18.99840316273,
}

atomic_no = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
}


'''
$ cat input/train.csv |cut -f 5 -d, |sort | uniq -c
 709416 1JHC
  43363 1JHN
1140674 2JHC
 378036 2JHH
 119253 2JHN
1510379 3JHC
 590611 3JHH
 166415 3JHN
-------------
4658147


$ cat input/test.csv |cut -f 5 -d, |sort | uniq -c
 380609 1JHC
  24195 1JHN
 613138 2JHC
 203126 2JHH
  64424 2JHN
 811999 3JHC
 317435 3JHH
  90616 3JHN
-------------
2505542
'''
