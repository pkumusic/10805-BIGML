from __future__ import division
from guineapig import *
import math
__Author__ = 'Music'

# supporting routines can go here

#always subclass Planner
class NB(Planner):
    # params is a dictionary of params given on the command line.
    # e.g. trainFile = params['trainFile']
    params = GPig.getArgvParams()
    data = ReadLines(params['trainFile']) | Map(by=lambda x:(1,[('a','Intuit'), ('se','Google')]))
    queries = ReadLines(params['trainFile']) | Map(by=lambda x: ('Intuit', 'Google'))

    # Flatten the data to get (id, company) tuples
    id_company = Flatten(data, by=lambda x: map(lambda y: (x[0],y[1]), x[1]))
    # Join every two companies with same id to get (id, company1, company2) tuples
    id_pairs = Join(Jin(id_company, by=lambda (id, company): id), \
                    Jin(id_company, by=lambda (id, company): id)) |\
        Map(by=lambda ((id1, company1), (id2, company2)):(id1, company1, company2))
    # Join company pairs with queries. Count the number of pairs fit to the queries.
    res = Join(Jin(id_pairs, by=lambda (id, company1, company2): (company1, company2)), \
               Jin(queries, by=lambda x:x)) | Map(by=lambda (x,y):y) |\
               Group(by=lambda x:x, reducingTo=ReduceToCount())




    #id_company = Flatten()

# always end like this
if __name__ == "__main__":
    NB().main(sys.argv) #

# supporting routines can go here
