"""
Author   : Abraham Flores
File     : bitcoin-model-USER-TEST.py
Language : Python 3.6
Created  : 5/9/2018
Edited   : 5/9/2018

San Digeo State University 
Computational Science Research Center (CSRC)
MTH 693b : Computational Partial Differential Equations
"""
#Make sure The test file and BITCOIN_MODEL.py are in the same path and can be found
from BITCOIN_MODEL import BlackScholes

if __name__ == "__main__":
    btc = "BTC_USD-Data.txt"
    btc60 = "btc60day.txt"
    bm = "SPY-Data.txt"
    rf = "US-10YR-BOND.txt"
    googl = "GOOGL.txt"
    trends = "bitcoin-trends.txt"
    trends60 = "trends_60day.txt"
    g_trends = "google-trends.txt"
    K = [3000,4000,5000,6000,7000,8000,9000,10000]
    n = 100
    p_iters = 500
    bs_iters = 5
    vol_iters = 25
    data = []
    
    test = BlackScholes(n,btc,bm,rf,trends,T=30,deg=35)
    for k in K:
        avg ,std = test.B_S_Time_Series(k,bs_iters,p_iters,vol_iters)
        print("STRIKE: "+str(k))
        print("Mean Call Price: " + str(avg))
        print("STD: "+str(std))
        data.append((k,avg,std))

