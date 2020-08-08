adress_data$mmse<-as.numeric(adress_data$mmse)
adress_data$label<-factor(adress_data$label,levels=c('Control','Dementia'))
adress_data$sex<-factor(adress_data$sex,levels=c(1,0))
adress_data$gender<-factor(adress_data$gender,levels=c('female','male'))
histo <- ggplot(adress_data, aes(mmse,fill=gender))


histo + geom_histogram(binwidth = 5) +labs(x ='mmse range', y = 'Frequency')
genbar2 <- ggplot(adress_data, aes(label, mmse, fill=label))
genbar2 + stat_summary(fun.y = mean, geom = 'bar') +stat_summary(fun.data = mean_cl_normal, geom ='errorbar', width = 0.1) + facet_wrap(~gender) +
labs(x ='group', y = 'mean_mmse')+theme(legend.position = 'none')
#cor.test(acoustic_feature$mfcc5_kurtosis, acoustic_feature$mmse, method ="pearson")
cor.test(acoustic_feature$mfcc5_kurtosis, acoustic_feature$mmse,method = "pearson")
acoustic_feature$mmse<-as.numeric(acoustic_feature$mmse)
adress_result_ngram_real_rbf$test<-as.numeric(adress_result_ngram_real_rbf$test)
adress_result_ngram_real_rbf$result<-as.numeric(adress_result_ngram_real_rbf$result)
cor.test(adress_result_ngram_real_rbf$test, adress_result_ngram_real_rbf$result)

