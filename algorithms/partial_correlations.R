library(PResiduals)
library(openxlsx)

#This function computes the partial Spearman's correlation coefficient (or its conditionnal
#version) for different columns of a data set df. The results are saved in an .xlsx file.

#For more details about the PResiduals package please refer to:
#     - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5949238/
#     - https://cran.r-project.org/web/packages/PResiduals/

correlations <- function(df, names_x, names_y, partial_names, link = "logit", pthresold = 0.05 ,
                         file_path , conditionnal = c(TRUE, FALSE), cond_name = NULL){
  
  ###################################################################################
  #df : dataframe , shape (n_samples , n_components_x + n_components_y + n_covariates)
  
  #names_x, names_y : lists of characters 
  #  names of the components
  
  #partial_names : list of characters
  #  names of the confounding factors we want to adjust 
  
  #link : character
  #  model name for the link function (see Presiduals documentation for more details)
  
  #pthresold : float <=1
  #   thresold for the significance of the correlation. We only consider the coefficient when the associated
  #   pvalue is <= pthresold
  
  #file_path : character
  #   file path for the xlsx file in which we will save the results 
  
  #conditionnal : boolean
  #   if TRUE, apply the conditional model
  
  #cond_name : character
  #   name of the categorical covariate we want to condition
  
  #RESULT:
  #     matrix of shape (n_components_x , n_components_y) if conditionnal = FALSE
  #     matrix of shape (n_components_x , n_components_y , n_categories) otherwise
  
  ###################################################################################
  
  start_time <- Sys.time()
  
  #1. Conditionnal model 
  if (conditionnal){
    
    #transform the conditionning variable into categorical variable (factor)
    df[ , cond_name] <- as.factor(df[ , cond_name])
    temp = unique(df[ , cond_name])
    
    result = array(data = 0 , dim = c(length(names_x),length(names_y) , length(temp)),
                   dimnames = list(names_x , names_y , temp))
    
    for (name_x in names_x){
      for (name_y in names_y){
        
        #create formula name_x | name_y ~ partial_name1 + partial_name2 + ...
        fm = as.formula(paste(name_x , paste(name_y , paste(partial_names , sep = '+') , sep = '~') , sep = '|'))
        
        #here we simply consider a categorical variable for the conditionning, the stratification method is used
        corr = conditional_Spearman(fm , conditional.by = cond_name ,data=df 
                                    , fit.x = "orm", fit.y = "orm", link.x = link , link.y = link
                                    , conditional.method = "stratification")
        i = 1
        for (cond in temp) {
          if (corr$est$est[i , "p"] <= pthresold) result[name_x , name_y , cond] = corr$est$est[i , "est"]
          i = i + 1
                           }
        
                             }     
                           }
    #save the result in an .xlsx file with multiple sheets (one for each category)
    save <- createWorkbook()
    for (cond in temp) {
      addWorksheet(save , as.character(cond))
      writeData(save , sheet = as.character(cond) , x=result[ , , cond] , rowNames = TRUE)
                       }
    saveWorkbook(save , file = file_path , overwrite = TRUE)
                  }
  
  #2. Partial model
  else {
    
    result = array(data = 0 , dim = c(length(names_x),length(names_y)),
                    dimnames = list(names_x , names_y))
    
    for (name_x in names_x){
      for (name_y in names_y){
        
        #create formula name_x | name_y ~ partial_name1 + partial_name2 + ...
        fm = as.formula(paste(name_x , paste(name_y , paste(partial_names , sep = '+') , sep = '~') , sep = '|'))
        
        corr = partial_Spearman(fm ,data=df , fit.x = "orm", fit.y = "orm",
                                link.x = link , link.y = link)
        
        if (corr$TS$TB$pval <= pthresold) result[name_x , name_y] = corr$TS$TB$ts

                             }
                            }
    #save the result in an .xlsx file
    write.xlsx(result, file = file_path , sheetName = "correlation" , overwrite = TRUE)
      }
  
  print(c("Running time estimate (in minutes): " , Sys.time() - start_time))
  
  return(result) 
}