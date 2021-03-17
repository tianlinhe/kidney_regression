import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns; sns.set()
import scipy.stats as stats
import statsmodels.api as sm

from set_path import *

class LR:
    def __init__(self, inputpep,inputpat,var,output1,outputresid,outputypred):
        self.inputpep=inputpep
        self.inputpat=inputpat
        self.output1=output1
        self.outputresid=outputresid
        self.outputypred=outputypred
        self.var=var
    def read(self):
        
        self.dfpat=pd.read_csv(self.inputpat,index_col=0)[self.var]
        
#         print (self.dfpat.columns.tolist())
#         print (self.dfpat.head())    
        self.dfpep=pd.read_csv(self.inputpep,index_col=0)
    
    def ols2(self):
        npep=len(self.dfpep.columns)
        npat=len(self.dfpat)
        nvar=len(self.var)
        
        
        
        # join peptide, covariates and y
        self.dfpep=self.dfpep.join(self.dfpat)
        
        self.df_out=pd.DataFrame(index=self.dfpep.columns.tolist()[:npep],
                           columns=['coef','std_err','rsquare','pvalue'])
        self.dfresid=pd.DataFrame(index=self.dfpep.index,
                           columns=self.dfpep.columns.tolist()[:npep])
        self.dfypred=self.dfresid.copy()
#         print (self.df_out.shape)
#         print (self.dfresid.shape)
        #print (self.dfpep.head())
        
        varlist=[npep+i for i in range(nvar-1)]
        #print (alist)
        
        self.dfpep.fillna(0, inplace=True)
        
        slp_ci=np.empty((npep,2))
        
        
        # y always the last column
        y=self.dfpep.iloc[:,-1]
        for i in range(npep):

            x=self.dfpep.iloc[:,[i]+varlist]

            x=sm.add_constant(x)
            lr=sm.OLS(y,x)
            res=lr.fit()
        
            # res.resid is the residues y.fit- ytrue of each peptides
            # res.params[1] is the coefficient of the 1st parameter, i.e. peptide
            # res.bse[1] is the pvalue of the 1st parameter, i.e. peptide
            # refer to https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.html
            
            self.dfresid.iloc[:,i]=res.resid
            self.dfypred.iloc[:,i]=res.fittedvalues
            self.df_out.iloc[i]=[res.params[1],res.bse[1],res.rsquared, res.pvalues[1]]
        
        # adjust p-values for multiple testing
        self.df_out=self.df_out.sort_values(['pvalue'], ascending = False) 
        Qvalue = [1]
        for row_num in range (len(self.df_out)):
        #important, q-value is the minimum FDR, so it always compares it with the preceding term, her the last term in the Qvalue list.
            q = min(len(self.df_out)*self.df_out.iloc[row_num]['pvalue']/(len(self.df_out)-row_num),Qvalue[-1])
            Qvalue.append(q)

        #to remove the first term (1) from the list of q-value
        Qvalue=Qvalue[1:]
        self.df_out['pvalue_adj'] = Qvalue
    
    def export_results(self):
        # (coefficient, standard error, p-value) of the peptide, and r-square of the model 
        self.df_out.to_csv(self.output1)
        # residues of the model
        self.dfresid.to_csv(self.outputresid)
        # fitted y values
        self.dfypred.to_csv(self.outputypred)

class residue:
    def __init__(self, inputresid,inputpep,inputpat,inputypred,var,output1,outputresid):
        self.inputypred=inputypred
        self.inputresid=inputresid
        self.inputpep=inputpep
        self.inputpat=inputpat
        self.output1=output1
        self.outputresid=outputresid
        self.var=var
    def read_resid(self):
        
        self.dfresid=pd.read_csv(self.inputresid,index_col=0)      
#         print (self.dfresid.shape)
#         print (self.dfresid.head())
    def errors_distribution(self,idx=0):
        """The errors are normally distributed"""
#         idx=self.dfresid.columns.index([idx])
#         print (self.dfresid.columns.tolist())

        X=self.dfresid.iloc[:,idx]
        st, p= stats.normaltest(X)
        fig, (ax0,ax1) = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
        fig.subplots_adjust(wspace=0.25)
        plt.suptitle('Distribution of residues in model '+self.dfresid.columns[idx]+
                     '\n $P_{norm}$='+'{0:0.3g}'.format(p))
        sns.distplot(X,ax=ax0)
        stats.probplot(X, dist="norm",plot=plt)

    def read_ypred_pep(self):
        self.dfypred=pd.read_csv(self.inputypred,index_col=0)
        self.dfpep=pd.read_csv(self.inputpep,index_col=0)
    def errors_0(self,idx):
        """The expected value of the errors is 0
        1. plot residues vs predictions
        2. plot residues vs independent variables (peptide)
        """
        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
        fig.subplots_adjust(wspace=0.25)
        ax[0].scatter(x=self.dfypred.iloc[:,idx],
                    y=self.dfresid.iloc[:,idx])
        ax[0].set_title("residues vs fitted")
        ax[0].set_ylabel("residues")
        ax[0].set_xlabel("fitted values\n lm(eGFR~ {}+ age + sex + CVD + diabetes)"\
                   .format(self.dfresid.columns[idx]))
        
        
        ax[1].scatter(x=self.dfpep.iloc[:,idx],
                    y=self.dfresid.iloc[:,idx])
        ax[1].set_title("residues vs peptide intensity")
        ax[1].set_ylabel("residues")
        ax[1].set_xlabel(self.dfresid.columns[idx])
    def variance_homogeneity(self,idx=0):
        """all errors have the same variance(homoskedasticity)
        should give a flat line
        """
        fig, ax= plt.subplots(1)#,figsize=(5,5))
#         fig.subplots_adjust(wspace=0.25)
        ax.scatter(x=self.dfypred.iloc[:,idx],
                    y=np.sqrt(abs(self.dfresid.iloc[:,idx])))
        ax.set_ylabel("sqrt(residues)")
        ax.set_xlabel("fitted values\n lm(eGFR~ {}+ age + sex + CVD + diabetes)"\
                   .format(self.dfresid.columns[idx]))
        
    def read_pateints(self):
        self.dfpat=pd.read_csv(self.inputpat,index_col=0)[self.var]
        self.dfresid=self.dfresid.join(self.dfpat)  
    
   
