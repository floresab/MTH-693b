"""
Author   : Abraham Flores
File     : BITCOIN_MODEL.py
Language : Python 3.6
Created  : 4/26/2018
Edited   : 5/9/2018

San Digeo State University 
Computational Science Research Center (CSRC)
MTH 693b : Computational Partial Differential Equations

This python file will determine the best-fit parameters for 
TGARCH volatility model for a specfic asset and compute the 
historical volatility for the benchmark and risk free asset. It 
aslo fits a drifted brownian motion forFor a benchmark and riskfree 
asset. It will then require a third asset along with its google search 
trend data. Once these have been obtained, using the time period 
set it will model out in days to that time. Report back a mean 
and standard deviation from a user set number of iterations. It will 
then use the Black-Scholes model with a FTCS scheme to solve the model 
out back to the start time and report back the price of the option. 
As a mean and standard devation from a user set number of iterations.

*Yes I realize I spelled volatility as volotility everywhere.

bitcoin-model-USER-TEST.py is a sample code of how to use this code
"""
import numpy as np
import scipy.optimize as optimize
from scipy import stats
from datetime import datetime

"""
function : Read_File

GOAL: 
    READ IN A FILE AND STORE THE COLUMNS IN A LIST OF LISTS
    
PARAMETERS:
    file_name: name of file to read
    num_header_lines: integer number of header lines before data
    cols: integer number of columns of data
    delim: how the columns are delimited
    
RETURN: 
    HEADER STRINGS IN A LIST , COLUMNS IN A LIST OF LISTS  
"""
def Read_File(file_name,delim=" "):
    with open(file_name) as file:
        #grab header
        header = file.readline().split()
        data = [[] for i in range(len(header))]
        #iterate over data
        for line in file:
            line_list=line.split(delim)
            #loop in parallel to append data to state
            for state, col in zip(data, line_list):
                state.append(col.strip())
    file.close()
    return header,data
        
"""
CLASS : Volotility

Data Members:
    STRING OF DATA FILE
    price_file
    benchmark_file
    risk_free_file
    #-----------------------------------
    periods = T #Time period to grab data from to compute H.V.
    vol_dates #Specific Date 
    vol_time  #Days after First Stored Date
    #-----------------------------------
    LIST OF ASSET VALUES 
    prices 
    benchmark
    risk_free
    #-----------------------------------
    #LOG DAILY RETURNS 
    returns
    benchmark_returns
    risk_free_returns
    #-----------------------------------
    LIST OF HISTORICAL VOLATILITIES
    volotility 
    benchmark_volotility 
    risk_free_volotility
    #-----------------------------------
    LAST COMPUTED HISTORICAL VOLATILIITY
    HV
    #-----------------------------------
    LIST OF RESIDUAL RETURNS
    residual_returns
    #-----------------------------------
    TGARCH OPTIMIZED PARAMETERS
    self.tgarch = results.x
    
    CALLABLE FUNCTION THAT EVALUATES WITH T,previous volatility for a fitted TGACH
    TGARCH_Fit = foo
    
Methods: 
    Get_Prices(self):                     READS AND ORGINIZES ASSET INFORMATION
    Returns(self):                        COMPUTES LOG RETURNS
    Historical_Volotility(self,T=30):     COMPUTES HV FOR EACH DAY USING T DAYS BEFORE
    Vol_SSE(self,parameters):             SSE EVALUATION FOR TGARCH MODEL
"""       
class Volotility:
    """
    intialization of Volotility class
    
    GOAL: 
        INTIALIZE PRICE DATA AND COMPUTE RELVENT INFORMATION, FIT DATA TO 
        TGARCH MODEL

    PARAMETERS: 
            price_file      : ASSET DATA FILE
            benchmark_fie   : BENCHMARK DATA FILE
            risk_free_file  : RISK FREE DATA FILE
            T : TIME PERIOD : DEFAULT = 30 DAYS
    """
    def __init__(self,price_file,benchmark_file,risk_free_file,T=30):
        self.price_file = price_file
        self.benchmark_file = benchmark_file
        self.risk_free_file = risk_free_file
        #-----------------------------------
        self.periods = T
        self.vol_dates = []
        self.vol_time = []
        self.prices = []
        self.benchmark = []
        self.risk_free = []
        #-----------------------------------
        self.Get_Prices() #Set Closing Prices
        #-----------------------------------
        self.returns = []
        self.benchmark_returns = []
        self.risk_free_returns = []
        #-----------------------------------
        self.Returns()   #Compute Daily Returns
        #-----------------------------------
        self.volotility = []
        self.benchmark_volotility = []
        self.risk_free_volotility = []
        #-----------------------------------
        self.Historical_Volotility(T) #Compute Daily Historical Volotility
        self.HV = self.volotility[-1]
        #-----------------------------------
        self.residual_returns = []
        #-----------------------------------
        self.Residual_Returns() #Compute Residual Returns
        
        #a0,a1,gamma,beta
        tgarch_guess = [1,1,1,1]
        
        results = optimize.minimize(self.Vol_SSE,tgarch_guess)
        if results.success:
            self.tgarch = results.x
            def foo(r,prev):
                a0,a1,gamma,beta = self.tgarch
                if r < 0:
                    return a0 + a1*r**2 + prev**2*beta
                else:
                    return a0 + (a1+gamma)*r**2 +prev**2*beta
            self.TGARCH_Fit = foo
                
        else:
            print(results)

        
    """
    Member Function : Get_Prices
    
    GOAL: Read in asset price data with times and dates. Store only matching
          dates accross the board. Store into approiate data members.
          
    """
    def Get_Prices(self):
        h1,price_data = Read_File(self.price_file,"\t")
        h2,benchmark_data = Read_File(self.benchmark_file,"\t")
        h3,risk_free_data = Read_File(self.risk_free_file,"\t")
        date_format = "%d-%b-%y"
        if len(price_data[0][0]) == 8:
            t0 = datetime.strptime("0"+price_data[0][0],date_format)
        else: 
            t0 = datetime.strptime(price_data[0][0],date_format)
        for date in price_data[0]:
            try:
                
                benchmark_index = benchmark_data[0].index(date)
                risk_free_index = risk_free_data[0].index(date)
                price_index = price_data[0].index(date)
                price = float(price_data[1][price_index].replace(",","").strip("\""))
                benchmark =  float(benchmark_data[1][benchmark_index].replace(",","").strip("\""))
                risk_free = float(risk_free_data[1][risk_free_index].replace(",","").strip("\""))
                self.prices.append(price)
                self.benchmark.append(benchmark)
                self.risk_free.append(risk_free)
                self.vol_dates.append(date)
                
                if len(date) == 8:
                    delta = datetime.strptime("0"+date,date_format) - t0
                    self.vol_time.append(delta.days+1)
                else:
                    delta = datetime.strptime(date,date_format) - t0
                    self.vol_time.append(delta.days+1)
                    
            except ValueError:
                continue
    """
    Member Function : Returns
    
    GOAL: Compute the Daily returns of a Stock and a Benchmark store in 
          appriate data members.
          
    """
    def Returns(self):
        for i in range(len(self.prices)-1):
            #Compute Returns for Our Stock
            R = np.log(self.prices[i+1]/self.prices[i])
            self.returns.append(R)
            
            #Compute Returns for Benchmark -- S and P 500
            R = np.log(self.benchmark[i+1]/self.benchmark[i])
            self.benchmark_returns.append(R)
            
            #Compute Returns for Risk Free -- Treasuary Bond
            R = np.log(self.risk_free[i+1]/self.risk_free[i])
            self.risk_free_returns.append(R)
        
    
    """
    Member Function : Historical_Volotility
    
    GOAL: Compute the Daily returns of a Stock and a Benchmark store in 
          appriate data members.
          
    """
    def Historical_Volotility(self,T=30):
        for i in range(len(self.returns[T:])+1):
            self.volotility.append(np.std(self.returns[i:i+T]))
            self.benchmark_volotility.append(np.std(self.benchmark_returns[i:i+T]))
            self.risk_free_volotility.append(np.std(self.risk_free_returns[i:i+T]))
    
    """
    Member Function : Residual_Returns
    
    GOAL: Compute the dailyresidual returns based from all three asset information
          and their corressponding volatility
          
    """
    def Residual_Returns(self,T=30):
        
        for i in range(len(self.returns[T:])):
            returns = self.returns[i:i+T]
            benchmark = self.benchmark_returns[i:i+T]
            beta = np.correlate(returns,benchmark)*(np.std(returns)/np.std(benchmark))
            alpha = \
            self.returns[i+T] - self.risk_free_returns[i+T] - \
            beta[0]*(self.benchmark_returns[i+T]-self.risk_free_returns[i+T])
            self.residual_returns.append(alpha)

    """
    function : TGARCH
    
    GOAL: EVALUTATE TGARCH MODEL GIVEN THE PARAMETERS,PREVIOUS VALUE AND RR

    PARAMETERS:
        params: = [a0,a1,gamma,beta] from TGARCH MODEL
        prev  : PREVIOUS VOLATILITY
        residual_return : residual return for previous day
        
    RETURN: predicted volatility
    """    
    def TGARCH(params,prev,residual_return):
        a0,a1,gamma,beta = params
        if residual_return < 0:
            return a0 + a1*residual_return**2 + prev**2*beta
        else:
            return a0 + (a1+gamma)*residual_return**2 +prev**2*beta 
    """
    Member Function : Vol_SSE
    
    GOAL: COMPUTE THE TOTAL SUM OF SQUARE ERRORS FOR A SPECIFIC SET 
          OF TGARCH PARAMETERS
          
    """
    def Vol_SSE(self,parameters):
        square_errors = 0
        for prev_,next_,rr in zip(self.volotility[:-1],self.volotility[1:],self.residual_returns[:-1]):
            vol = Volotility.TGARCH(parameters,prev_,rr)
            square_errors += (vol - next_**2)**2
        return square_errors
  
"""
CLASS : GoogleTrends

Data Members: 
    trends_file  : DATA FILE
    trends       : TRENDING VALUE [0,100]
    trend_dates  : SPECIFIC DATES FOR VALUE
    trend_time   : TIME IN DAYS AFTER FIRST DATE
    #-----------------------------------
    degrees      : DEGREES TO FIT CHEBYSHEV POLYNOMIAL
    Interpolate  : CHEBYSHEV POLYNOMIAL FIT TO GOOGLE TREND DATA WITH X=TREND_TIME
    #-----------------------------------
    Extrapolate  : EXTRAPOLATED FIT TO GOOGLE TRENDS USING T DAYS PREVIOUSLY
    
Methods: 
    blah
"""
class GoogleTrends:
    """
    INTIALIZE GOOGLE TREND CLASS

    PARAMETERS:
        treds_file     : name data file
        degrees_of_fit : degrees to fit chebyshev polynomial
        T : time_period to fit extroplated data
    """    
    def __init__(self,trends_file,degrees_of_fit=35,T=30):
        self.trends_file = trends_file
        self.trends = []
        self.trend_dates = []
        self.trend_time = []
        #-----------------------------------
        self.Get_Trends()
        #-----------------------------------
        self.degrees = degrees_of_fit
        #Callable Function
        self.Interpolate = \
        np.polynomial.chebyshev.Chebyshev.fit(self.trend_time,self.trends,self.degrees)
        #-----------------------------------
        buffer = 5
        days = np.array([t for t in range(-T,-buffer+1)])
        t = days + self.trend_time[-1]
        trends_ =  np.log(self.Interpolate(t))
        values =  stats.linregress(days,trends_)
        b = values[0]
        A = np.exp(values[1])
        #-----------------------------------
        def foo(t):
            return A*np.exp(t*b)
        #-----------------------------------
        #Callable Function
        self.Extrapolate = foo

    """
    Member Function :  Get_Trends(self)
    
    GOAL: READ IN TREND DATA WITH DATES, WRITE TO TREND,DATES,TIME LIST
          
    """  
    def Get_Trends(self):
        header,data = Read_File(self.trends_file,"\t")
        self.trend_dates = data[0]
        self.trends = [float(x) for x in data[1]]
        date_format = "%m/%d/%Y"
        t0_list = str(data[0][0]).split("/")
        for i in range(2):
            if len(t0_list[i]) != 2:
                t0_list[i] = "0" + t0_list[i]
                
        t0 = datetime.strptime("/".join(t0_list),date_format)
        for date in data[0]:
            date_list = str(date).split("/")
            for i in range(2):
                if len(date_list[i]) != 2:
                    date_list[i] = "0"+date_list[i]
            delta = datetime.strptime("/".join(date_list),date_format) - t0
            self.trend_time.append(delta.days)
        
    
"""
CLASS : Stock
    -- Inheirited 
        --Volotility 
        --GoogleTrends

Data Members: 
    
    Volotility   : INHERITANCE
    GoogleTrends : INHERITANCE
    #-----------------------------------
    benchmark_mu : fitted drift parameter
    Benchmark_Extrapolate :   Callable function for fitted model
    #-----------------------------------
    risk_free_mu : fitted drift parmeter
    Risk_Free_Extrapolate : Callable function for fitted model
    #-----------------------------------
    self.stock_mu = fitted drift parameter
    self.HYPE = fitted HYPE parameter
    self.HYPE_Extrapolate = Callable function for HYPE model
    
Methods: 
    Benchmark_SSE(self,mu)    :  SUM OF SQUARE ERRORS FOR BENCHMARK MODEL
    Risk_Free_SSE(self,mu)    :  SUM OF SQUARE ERRORS FOR RISK FREE MODEL
    Stock_SSE(self,mu)        :  SUM OF SQUARE ERRORS FOR STOCK MODEL
    Hype_SSE(self,parameters) :  SUM OF SQUARE ERRORS FOR HYPE MODEL
    Time_Series(self,iters=25):  TIME SERIES TO T FOR PRICE OF STOCK
"""
class Stock(Volotility,GoogleTrends):
    """
    INTIALIZATION OF STOCK CLASS
    
    GOAL: OPTIMIZE MODELS WITH COBYLA METHOD GENERATE CALLABLE EXTRAPOLATION 
          FUNCTIONS FOR USER. INFORM USER IF OPTIMAZATION FAILED

    PARAMETERS:
        price_file      : ASSET DATA FILE
        benchmark_fie   : BENCHMARK DATA FILE
        risk_free_file  : RISK FREE DATA FILE
        trends_file     : GOOGLE TRENDS DATA FILE
        T : TIME PERIOD : DEFAULT = 30 DAYS
        deg             : CHEBYSHEV DEGREES : DEFAULT = 30 DAYS
        
    """
    def __init__(self,price_file,benchmark_file,risk_free_file,trends_file,T=30,deg=35):
        Volotility.__init__(self,price_file,benchmark_file,risk_free_file,T)
        GoogleTrends.__init__(self,trends_file,deg,T)
        #-----------------------------------
        self.benchmark_mu = 1
        self.benchmark_sigma = self.benchmark_volotility[-1]
        #-----------------------------------
        results = optimize.minimize(self.Benchmark_SSE,10**(-1),method="COBYLA")
        if results.success:
            self.benchmark_mu = results.x
        #-----------------------------------
            def foo1(prev):
                epsilon = np.random.normal(0, 1.0, 1)
                return prev*np.exp(\
            (self.benchmark_mu-self.benchmark_sigma**2/2) + self.benchmark_sigma*epsilon)
        #-----------------------------------
            self.Benchmark_Extrapolate = foo1
        #-----------------------------------
        else:
            print("Benchmark Optimaztion Failed")
            print(results)
        #-----------------------------------
        self.risk_free_mu = 1
        self.risk_free_sigma = self.risk_free_volotility[-1]
        #-----------------------------------
        results = optimize.minimize(self.Risk_Free_SSE,10**(-1),method="COBYLA")
        if results.success:
            self.risk_free_mu = results.x
            def foo2(prev):
                epsilon = np.random.normal(0, 1.0, 1)
                return prev*np.exp(\
            (self.risk_free_mu-self.risk_free_sigma**2/2) + self.risk_free_sigma*epsilon)
        #-----------------------------------
            self.Risk_Free_Extrapolate = foo2
        #-----------------------------------
        else:
            print("Risk Free Optimaztion Failed")
            print(results)
        #-----------------------------------
        self.stock_mu = .1
        #-----------------------------------
        results = optimize.minimize(self.Stock_SSE,0.1,method="L-BFGS-B")
        if results.success:
            self.stock_mu = results.x[0]
        #-----------------------------------
        else:
            print("Stock Optimaztion Failed")
            print(results)
        #-----------------------------------
        self.HYPE = .1
        #-----------------------------------
        results = optimize.minimize(self.Hype_SSE,10,method="L-BFGS-B")
        if results.success:
            self.HYPE = results.x[0]
            def foo3(prev,sigma,t):
                epsilon = np.random.normal(0, 1.0, 1)
                return prev*np.exp(self.stock_mu*(1+self.HYPE*self.Extrapolate(t))-sigma**2/2 + sigma*epsilon)
            
            self.HYPE_Extrapolate = foo3
        else:
            print("Hype Optimaztion Failed")
            print(results)
    
    """
    function : Mean_Field_Model
    
    GOAL: 
        EVALUATE MODEL FOR SINGLE INSTANCE GIVEN MU
        
    PARAMETERS:
        mu: DRIFT 
        previous: PREVIOUS VALUE OF ASSET
        sigma : VOLATILITY
        dt: CHANGE IN TIME TO NEXT DATE
        
    RETURN:
        PRICE AT THE NEXT TIME STEP (t+dt)
    """
    def Mean_Field_Model(mu,previous,sigma,dt):
        return previous*np.exp(dt*(mu-sigma**2/2))
    
    """
    Member Function :  Benchmark_SSE(self,mu)
    
    GOAL: EVALUTATE SUM OF SQUARE ERRORS FOR BROWNIAN MOTION MODEL
        
    RETURN: SSE
    """  
    def Benchmark_SSE(self,mu):
        square_errors = 0
        for prev_price,next_price,vol\
        in zip(self.benchmark[self.periods-1:-1],self.benchmark[self.periods:],self.benchmark_volotility):
            model = Stock.Mean_Field_Model(mu,prev_price,vol,1)
            square_errors += (model - next_price)**2
        
        return square_errors
    
    """
    Member Function :  Risk_Free_SSE(self,mu)
    
    GOAL: EVALUTATE SUM OF SQUARE ERRORS FOR BROWNIAN MOTION MODEL
        
    RETURN: SSE
    """  
    def Risk_Free_SSE(self,mu):
        square_errors = 0
        for prev_price,next_price,vol\
        in zip(self.risk_free[self.periods-1:-1],self.risk_free[self.periods:],self.risk_free_volotility):
            model = Stock.Mean_Field_Model(mu,prev_price,vol,1)
            square_errors += (model - next_price)**2
        
        return square_errors
    
    """
    Member Function :  Stock_SSE(self,mu)
    
    GOAL: EVALUTATE SUM OF SQUARE ERRORS FOR BROWNIAN MOTION MODEL
        
    RETURN: SSE
    """ 
    def Stock_SSE(self,mu):
        square_errors = 0
        for prev_price,next_price,vol\
        in zip(self.prices[self.periods-1:-1],self.prices[self.periods:],self.volotility):
            model = Stock.Mean_Field_Model(mu,prev_price,vol,1)
            square_errors += (model - next_price)**2
        
        return square_errors
    """
    function : Mean_Hype_Model
    
    GOAL: 
        EVALUATE MODEL FOR SINGLE INSTANCE GIVEN ALL PARAMETERS
        
    PARAMETERS:
        parameters : Hype parameter
        previous: the current price of bitcoin
        mu: drift
        sigma : volotility
        dt: time step
        
    RETURN:
        PRICE AT THE NEXT TIME STEP (t+dt)
    """
    def Mean_Hype_Model(parameters,previous,mu,sigma,dt,trend):
        hype = parameters
        return previous*np.exp(dt*(mu*(1+hype*trend)-sigma**2/2))
    
    """
    Member Function :  Stock_SSE(self,mu)
    
    GOAL: EVALUTATE SUM OF SQUARE ERRORS FOR BROWNIAN MOTION MODEL
        
    RETURN: SSE
    """ 
    def Hype_SSE(self,parameters):
        square_errors = 0
        for prev_price,next_price,vol,t\
        in zip(self.prices[self.periods-1:-1]\
               ,self.prices[self.periods:]\
               ,self.volotility\
               ,self.vol_time[self.periods-1:-1]):

            if t > self.trend_time[-1]-7:
                trend = self.Extrapolate(t - self.trend_time[-1])
            else:
                trend = self.Interpolate(t)
                
            model = Stock.Mean_Hype_Model(parameters,prev_price,self.stock_mu,vol,1,trend)
            square_errors += (model - next_price)**2

        return square_errors
    """
    Helper function for Time Series
    """
    def Returns_(p):
        R = []
        for i in range(len(p)-1):
            R.append((np.log(p[i+1]/p[i]))[0])
        return R
    """
    Helper function for Time Series
    """
    def Residual_Returns_(R,Rb,Rf,vol_benchmark):
        rr = []
        beta = np.correlate(R,Rb)*(np.std(R)/vol_benchmark)
        for i in range(len(R)):
            alpha = R[i] - Rf[i] - beta[0]*(Rb[i]-Rf[i])
            rr.append(alpha)
        return rr
    
    """
    Member Function :  Time_Series(self,iters)
    
    GOAL: SIMULATE STOCK PRICE OUT TO T=PERIODS
        
    RETURN: LISTS , TIME, PRICE, VOLATILITY
    """ 
    def Time_Series(self,iters=25):
        time = [t for t in range(self.periods)]
        prices = [self.prices[-1]]
        HV = self.volotility[-1]
        volotility = np.zeros(self.periods)
        volotility[0] = HV
        benchmark = [self.benchmark[-1]]
        risk_free = [self.risk_free[-1]]
        
        for day in range(self.periods-1):
            benchmark.append(self.Benchmark_Extrapolate(benchmark[day]))
            risk_free.append(self.Risk_Free_Extrapolate(risk_free[day]))
            prices.append(self.HYPE_Extrapolate(prices[day],HV,day+1))
        BmR = Stock.Returns_(benchmark)
        RFR = Stock.Returns_(risk_free)

        for i in range(iters):
            returns = Stock.Returns_(prices)
            residual_returns = Stock.Residual_Returns_(returns,BmR,RFR,self.benchmark_sigma)
            for day in range(self.periods-1):
                volotility[day+1] = np.sqrt(self.TGARCH_Fit(residual_returns[day],volotility[day]))
                prices[day+1] = self.HYPE_Extrapolate(prices[day],volotility[day+1],day+1)

        return time,prices,volotility

"""
CLASS : BlackScholes
    -- Inheirited 
        Stock
            --Volotility 
            --GoogleTrends

Data Members: 
    STOCK : INHERITANCE
        Volotility   : INHERITANCE
        GoogleTrends : INHERITANCE
    #-----------------------------------
    grid_steps : Number of Discritization points in price grid
    time_steps : Number of Discritization points in spatial grid
    time       : Time in Days to T
    #-----------------------------------
    ***INTIALIZED IN Black_Scholes_Time_Series***
    strike_price : GIVEN STRIKE PRICE
    call_list    : LIST OF PREDICTED CALL PRICES FOR OUR STOCK PRICE AT t=0
    #-----------------------------------
    mean_prices  : MEAN PRICE FOR RUNING STOCK TIME SERIES FOR ASSET VALUE
    std_prices   : STD PRICE FOR RUNING STOCK TIME SERIES FOR ASSET VALUE
    mean_vols    : MEAN VOLATILITY FOR RUNING STOCK TIME SERIES FOR ASSET VALUE
    std_vols     : STD VOLATILITY FOR RUNING STOCK TIME SERIES FOR ASSET VALUE
    #-----------------------------------
    mean : MEAN PRICE AT t = T FROM THE STOCK TIME SERIES
    std  : STD PRICE AT t = T FROM THE STOCK TIME SERIES
    vol  : MEAN VOLATILITY AT t = T FROM THE STOCK TIME SERIES
    Smax : MAXIMUM ASSET VALUE MODEL PREDICTS AT TIME T
    S_grid : DISCRETIZED ASSET VALUE GRID [K,SMAX]
    dS     : CHANGE IN ASSET VALUE BETWEEN GRID POINTS
    index  : INDEX OF MEAN ASSET VALUE -- BOOKEEPING FOR THE ASSET VALUE WE CARE ABOUT
    dt     : ANNUALIZED CHANGE IN TIME BETWEEN ASSET EVALUATION
    #-----------------------------------
    risk_free_rate : RISK FREE INTRESTED RATE ... ANNUALIZED
    option_price   : PREDICTED VALUE OF THE CALL PRICE
    #-----------------------------------
            
    
Methods: 
    Black_Scholes_Time_Series(self,K,bs_iters=5,p_iters=100,vol_iters=25):
    Set_Matrix(self,alpha,t):
"""
class BlackScholes(Stock):
    """
    INTIALIZATION OF BlackScholes CLASS
    
    GOAL: INTIALIZE STOCK CLASS AND ASSIGN DATA MEMBERS VALUES

    PARAMETERS:
        n               : ASSET GRID STEPS
        price_file      : ASSET DATA FILE
        benchmark_file   : BENCHMARK DATA FILE
        risk_free_file  : RISK FREE DATA FILE
        trends_file     : GOOGLE TRENDS DATA FILE
        T : TIME PERIOD : DEFAULT = 30 DAYS
        deg             : CHEBYSHEV DEGREES : DEFAULT = 30 DAYS
        
    """
    def __init__(self,n,price_file,benchmark_file,risk_free_file,trends_file,T=30,deg=35):

        self.grid_steps =  n
        self.time_steps = T
        self.time = [t for t in range(T)]
        #-----------------------------------
        Stock.__init__(self,price_file,benchmark_file,risk_free_file,trends_file,T,deg)
        print("Successful Optimization")
        #-----------------------------------
        
    """
    Member Function :  Black_Scholes_Time_Series(self,K,bs_iters=5,p_iters=100,vol_iters=25)
    
    GOAL: MODEL CALL PRICE GIVEN STRIKE PRICE K
    
    PARAMETERS:
        K          : STRIKE PRICE
        bs_iters   : ITERATIONS FOR BLACK SCHOLES MODEL       : DEFAULT = 5
        p_iters    : ITERATIONS FOR PRICE/HYPE MODEL          : DEFAULT = 100
        vol_iters  : ITERATIONS FOR HYPE/TGARCH FEEDBACK LOOP : DEFAULT = 25

    RETURN: MEAN AND STD OF PREDICTED CALL VALUE
    """ 
    def Black_Scholes_Time_Series(self,K,bs_iters=5,p_iters=100,vol_iters=25):
        self.strike_price = K
        self.call_list = []
        for k in range(bs_iters):
            print("Begining Black-Scholes Loop: " + str(k+1) + " of " + str(bs_iters))
            price_container = [[] for x in range(self.time_steps)]
            vol_container = [[] for x in range(self.time_steps)]
            for i in range(p_iters):
                time,prices,vol = self.Time_Series(vol_iters)
                for j in range(len(prices)):
                    price_container[j].append(prices[j])
                    vol_container[j].append(vol[j])
            #-----------------------------------
            self.mean_prices = [np.mean(p) for p in price_container]
            self.std_prices = [np.std(p) for p in price_container]
            self.mean_vols = [np.mean(vol) for vol in vol_container]
            self.std_vols = [np.std(vol) for vol in vol_container]
            
            self.mean_prices.reverse()
            self.std_prices.reverse()
            self.mean_vols.reverse()
            self.std_vols.reverse()
            self.time.reverse()
            #-----------------------------------   
            if self.mean_prices[0] < self.strike_price:
                print("WARNING MEAN PRICE AT EXPIRY IS LESS THAN STRIKE, Mean Price at T: ",self.mean_prices[0])
            #-----------------------------------
            self.mean = self.mean_prices[0]
            self.std = self.std_prices[0]
            self.vol = self.mean_vols[0]
            self.Smax = self.mean+self.std
            self.S_grid = np.linspace(self.strike_price,self.Smax,self.grid_steps)
            self.dS = (self.mean_prices[0] + self.std_prices[0] - self.strike_price)/(self.grid_steps-1)
            self.index = int((self.mean-self.strike_price)/self.dS)
            self.dt =  1.0/365.0
            intial = self.S_grid - self.strike_price
            
            self.risk_free_rate = ((self.risk_free_mu - self.risk_free_sigma**2/2)+1)**(365/self.periods) - 1.0
            alpha = 0#(22/self.time_steps)*10**(-3.2)
            self.option_price = intial
            for t in range(self.time_steps):
                black_scholes = self.Set_Matrix(alpha,t)
                #self.option_price *= np.exp(-alpha*t)
                self.option_price = np.matmul(black_scholes,self.option_price)
            
            self.option_price *= np.exp(alpha*self.periods)
            self.call_list.append((self.option_price[self.index-1] + self.option_price[self.index] + self.option_price[self.index+1])/3)
            
        return sum(self.call_list)/len(self.call_list) , np.std(self.call_list)
    
    """
    Member Function :  Set_Matrix(self,alpha,t)
    
    GOAL: BUILD FTCS MATRIX FOR BLACK-SCHOLES EQUATION 
    
    PARAMETERS:
        alpha  : CHANGE OF VARIABLE VALUE
        t      : TIME FOR GOOGLE TREND EXTRAPOLATE AND DATA

    RETURN: MATRIX
    """ 
    def Set_Matrix(self,alpha,t):
        FTCS = np.zeros((self.grid_steps,self.grid_steps))
        sigma = self.mean_vols[t]
        r = self.risk_free_rate + alpha
        h = (np.log(self.mean_prices[t] + self.std_prices[t]) - 1.0)/(self.grid_steps-1)
        k = self.dt
        mu_prime = self.stock_mu*(1+self.Extrapolate(self.periods-t)*self.HYPE) - sigma**2/2
        lambda_ = k/h
        
        #off Boundary
        for j in range(1,self.grid_steps-1): 
            
            FTCS[j][j-1] = lambda_*(mu_prime - sigma**2/(2*h))
            FTCS[j][j] = 1.0 + k*r + sigma**2*lambda_/h
            FTCS[j][j+1] = lambda_*(-mu_prime - sigma**2/(2*h))
            
        #On Boundary
        #Constant Zero
        FTCS[0][0] = 1.0
        
        #Ghost Point = Last Point
        FTCS[-1][-1] = 1.0 + k*r + sigma**2*lambda_/h + lambda_*(-mu_prime - sigma**2/(2*h))
        FTCS[-1][-2] = lambda_*(mu_prime - sigma**2/(2*h))

        return FTCS