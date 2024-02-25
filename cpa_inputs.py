rates = [
    'DB(CBN,DEM,10,MIDYLD)',
    'DB(CBN,ITL,10,MIDYLD)',
    'DB(CBN,USD,10,MIDYLD)'
]
uncertainty = [
    #'DB(QA,.VIX,PRICE,CLOSE)',
    'DB(COV,VOLSWAPTION,EUR,05Y,06M,PAYER,PREM)',
    'sum(DB(NEO-US, eur/vol/index/swaption/3mx2y/daily))'
    ]
politics = []
govt_finances = [
    'DB(JPMAQS,ITL_GGNDGDPRATIO_NSA,value)',
    'DB(JPMAQS,ITL_GGPBGDPRATIO_NSA,value)',
    'DB(JPMAQS,ITL_GGOBGDPRATIO_NSA,value)',
    'DB(JPMAQS,ITL_GGSBGDPRATIO_NSA,value)',
    'DB(JPMAQS,DEM_GGNDGDPRATIO_NSA,value)',
    'DB(JPMAQS,DEM_GGPBGDPRATIO_NSA,value)',
    'DB(JPMAQS,DEM_GGOBGDPRATIO_NSA,value)',
    'DB(JPMAQS,DEM_GGSBGDPRATIO_NSA,value)'
    ]
macro_trends = [
    'DB(JPMAQS,USD_PCREDITGDP_SA,value)',
    'DB(JPMAQS,USD_MBASEGDP_SA,value)',
    #'DB(EDG,D:MSEROP$$,PI)',
    #'DB(QA,.SPX,PRICE,CLOSE)',
    'DB(JPMAQS, ITL_RGDP_SA_P1Q1QL4_20QMA, value)',
    'DB(JPMAQS, DEM_RGDP_SA_P1Q1QL4_20QMA, value)',
    'DB(JPMAQS, USD_RGDP_SA_P1Q1QL4_20QMA, value)'
    ]
monetary_policy = [
    'DB(FROLL,3M,MIDYLD)',
    'DB(MTE,dem/gov/pc/unadj//01Y/rate)'
    ]
price_data = [
    'DB(EDG,D:MSEROP$$,PI)',
    'DB(QA,.SPX,PRICE,CLOSE)',
]


cat_dict = {
    'DB(CBN,DEM,10,MIDYLD)': '10-year Germany',
    'DB(MTE,dem/gov/pc/unadj//01Y/rate)': '1-year Germany',
    'DB(CBN,ITL,10,MIDYLD)': '10-year Italy',
    'DB(CBN,USD,10,MIDYLD)': '10-year US',
    'DB(FROLL,3M,MIDYLD)': '3-month FRA',
    'DB(JPMAQS,ITL_GGNDGDPRATIO_NSA,value)': 'Italy Debt',
    'DB(JPMAQS,ITL_GGPBGDPRATIO_NSA,value)': 'Italy Primary Balance',
    'DB(JPMAQS,ITL_GGOBGDPRATIO_NSA,value)': 'Italy Overall Balance',
    'DB(JPMAQS,ITL_GGSBGDPRATIO_NSA,value)': 'Italy Structural Balance',
    'DB(JPMAQS,DEM_GGNDGDPRATIO_NSA,value)': 'Germany Debt',
    'DB(JPMAQS,DEM_GGPBGDPRATIO_NSA,value)': 'Germany Primary Balance',
    'DB(JPMAQS,DEM_GGOBGDPRATIO_NSA,value)': 'Germany Overall Balance',
    'DB(JPMAQS,DEM_GGSBGDPRATIO_NSA,value)': 'Germany Structural Balance',
    'DB(JPMAQS,USD_PCREDITGDP_SA, value)': 'US Private Credit',
    'DB(JPMAQS,USD_MBASEGDP_SA,value)': 'US Monetary Base',
    'DB(JPMAQS, ITL_RGDP_SA_P1Q1QL4_20QMA, value)': 'Italy GDP',
    'DB(JPMAQS, DEM_RGDP_SA_P1Q1QL4_20QMA, value)': 'Germany GDP',
    'DB(JPMAQS, USD_RGDP_SA_P1Q1QL4_20QMA, value)': 'US GDP',
    'DB(COV,VOLSWAPTION,EUR,05Y,06M,PAYER,PREM)': 'EUR 5Y 1M payer vol',
    'sum(DB(NEO-US, eur/vol/index/swaption/3mx2y/daily))': 'EUR 2Y 3M payer vol',
    }

cat_dict_short = {
    'DB(CBN,DEM,10,MIDYLD)': '10y GE',
    'DB(MTE,dem/gov/pc/unadj//01Y/rate)': '1y GE',
    'DB(CBN,ITL,10,MIDYLD)': '10y IT',
    'DB(CBN,USD,10,MIDYLD)': '10y US',
    'DB(FROLL,3M,MIDYLD)': '3-month FRA',
    'DB(JPMAQS,ITL_GGNDGDPRATIO_NSA,value)': 'IT Debt',
    'DB(JPMAQS,ITL_GGPBGDPRATIO_NSA,value)': 'IT PB',
    'DB(JPMAQS,ITL_GGOBGDPRATIO_NSA,value)': 'IT OB',
    'DB(JPMAQS,ITL_GGSBGDPRATIO_NSA,value)': 'IT SB',
    'DB(JPMAQS,DEM_GGNDGDPRATIO_NSA,value)': 'GE Debt',
    'DB(JPMAQS,DEM_GGPBGDPRATIO_NSA,value)': 'GE PB',
    'DB(JPMAQS,DEM_GGOBGDPRATIO_NSA,value)': 'GE OB',
    'DB(JPMAQS,DEM_GGSBGDPRATIO_NSA,value)': 'GE SB',
    'DB(JPMAQS,USD_PCREDITGDP_SA,value)': 'US Pr Credit',
    'DB(JPMAQS,USD_MBASEGDP_SA,value)': 'US MB',
    'DB(JPMAQS, ITL_RGDP_SA_P1Q1QL4_20QMA, value)': 'IT GDP',
    'DB(JPMAQS, DEM_RGDP_SA_P1Q1QL4_20QMA, value)': 'GE GDP',
    'DB(JPMAQS, USD_RGDP_SA_P1Q1QL4_20QMA, value)': 'US GDP',
    'DB(COV,VOLSWAPTION,EUR,05Y,06M,PAYER,PREM)': 'EUR 5m5y vol',
    'sum(DB(NEO-US, eur/vol/index/swaption/3mx2y/daily))': 'EUR 3m2y vol',
    }



