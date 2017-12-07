# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:07:05 2017

@author: hecto
"""

    #building linear predictive model for imputation using backward elimination
    col_index = list((range(0, X_temp_train.shape[1])))
    backwardEliminationComplete = False
    X_opt = X_temp_train
    
    print(X_temp_train[0:5])
    print(X_temp_train.shape)
    print(X_opt[0:5])
    print(X_opt.shape)

            
    while (backwardEliminationComplete == False):
        regressor_OLS = sm.OLS(endog=y_temp_train, exog=X_opt).fit()
        isThereAPvalueGreaterThan5 = len(regressor_OLS.pvalues[regressor_OLS.pvalues > 0.05]) > 0
        if(isThereAPvalueGreaterThan5):
            #removes highest p-value
            cols = regressor_OLS.pvalues[regressor_OLS.pvalues != regressor_OLS.pvalues.max()]
            print(cols)
            X_opt = X_opt[cols]
        else:
            backwardEliminationComplete = True
        
    print(X_temp_test)
    print(y_temp_test.sort_values().values)
    y_pred_age = regressor_OLS.predict(X_temp_test[X_opt.columns])
#    print(classification_report(y_temp_test, y_pred_age))
    
    plt.plot(y_temp_test.sort_values().values, color='r')
    plt.plot(y_pred_age.sort_values().values, color='b')
    plt.show()
    
    print(regressor_OLS.rsquared)
    
    
    X["x0"] = pd.Series(np.ones(X.shape[0]).astype(int), index=X.index)
    X["Age"] = X["Age"].fillna(age_regressor.predict(X_temp_test[X_temp_train_poly.columns]))
