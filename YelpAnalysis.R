rm(list=ls())   #Clears the workspace
library(dplyr)
library(stringr)   #For partial match function (str_detect)
library(caret)
library(rpart)
library(rpart.plot)
library(e1071)   #For SVM

#install.packages("MLmetrics")
library(MLmetrics)


# read csv file
yelp <- read.csv("C:/Users/yosuk/Desktop/YelpAnalysis/CleanedData/yelp_business_all_clean3.csv", stringsAsFactors = FALSE)

# select open restaurants 
#yelpRes <- subset(yelp, str_detect(yelp$categories, "Restaurants") & yelp$is_open == 1)
yelpRes <- subset(yelp, str_detect(yelp$categories, "Restaurants"))


#table(yelpRes$attributes.BusinessParking)
#yelpRes$attributes.BusinessParking <- ifelse(yelpRes$attributes.BusinessParking %in% c("Garage","Lot","Street","Valet","Validated"), "True", yelpRes$attributes.BusinessParking)


# filter null values
yelpResClean <-subset(yelpRes, attributes.RestaurantsPriceRange2 != "NULL" & 
                          attributes.BusinessAcceptsCreditCards != "NULL" &
                          attributes.NoiseLevel != "NULL" &
                          attributes.RestaurantsAttire != "NULL" &
                          attributes.RestaurantsDelivery != "NULL" &
                          attributes.RestaurantsTakeOut != "NULL" &
                          attributes.GoodForKids != "NULL" &
                          attributes.RestaurantsGoodForGroups != "NULL" &
                          attributes.RestaurantsReservations != "NULL" &
                          attributes.OutdoorSeating != "NULL" &
                          attributes.HasTV != "NULL" &
                          attributes.WiFi != "NULL" &
                          attributes.BusinessParking != "NULL")
                        

# data understanding
str(yelpResClean)
table(yelpResClean$state)
hist(yelpResClean$stars)
hist(yelpResClean$review_count)


# convert character into num
yelpResClean$stars <- as.numeric(yelpResClean$stars)
yelpResClean$review_count <- as.numeric(yelpResClean$review_count)

# convert character into factor
yelpResClean$attributes.RestaurantsPriceRange2 <- as.factor(yelpResClean$attributes.RestaurantsPriceRange2)
yelpResClean$attributes.BusinessAcceptsCreditCards <- as.factor(yelpResClean$attributes.BusinessAcceptsCreditCards)
yelpResClean$attributes.NoiseLevel <- as.factor(yelpResClean$attributes.NoiseLevel)
yelpResClean$attributes.RestaurantsAttire <- as.factor(yelpResClean$attributes.RestaurantsAttire)
yelpResClean$attributes.RestaurantsDelivery <- as.factor(yelpResClean$attributes.RestaurantsDelivery)
yelpResClean$attributes.RestaurantsTakeOut <- as.factor(yelpResClean$attributes.RestaurantsTakeOut)
yelpResClean$attributes.GoodForKids <- as.factor(yelpResClean$attributes.GoodForKids)
yelpResClean$attributes.RestaurantsGoodForGroups <- as.factor(yelpResClean$attributes.RestaurantsGoodForGroups)
yelpResClean$attributes.RestaurantsReservations <- as.factor(yelpResClean$attributes.RestaurantsReservations)
yelpResClean$attributes.OutdoorSeating <- as.factor(yelpResClean$attributes.OutdoorSeating)
yelpResClean$attributes.HasTV  <- as.factor(yelpResClean$attributes.HasTV)
yelpResClean$attributes.WiFi  <- as.factor(yelpResClean$attributes.WiFi)
yelpResClean$attributes.BusinessParking  <- as.factor(yelpResClean$attributes.BusinessParking)


# create columnList
columnList <- c("stars", "attributes.RestaurantsPriceRange2", "attributes.BusinessAcceptsCreditCards",
                "attributes.NoiseLevel", "attributes.RestaurantsAttire",
                "attributes.RestaurantsDelivery", "attributes.RestaurantsTakeOut",
                "attributes.GoodForKids", "attributes.RestaurantsGoodForGroups",
                "attributes.RestaurantsReservations", "attributes.OutdoorSeating",
                "attributes.HasTV", "attributes.WiFi", "attributes.BusinessParking") 

# create separate dataframe
yelpResCol <- select(yelpResClean, columnList)
str(yelpResCol)


########## Linear Regression
Reg <- lm(stars~attributes.RestaurantsPriceRange2+
           attributes.BusinessAcceptsCreditCards+
           attributes.NoiseLevel+
           attributes.RestaurantsAttire+
           attributes.RestaurantsDelivery+
           attributes.RestaurantsTakeOut+
           attributes.GoodForKids+
           attributes.RestaurantsGoodForGroups+
           attributes.RestaurantsReservations+
           attributes.OutdoorSeating+
           attributes.HasTV+
           attributes.WiFi+
           attributes.BusinessParking,
           data = yelpResCol)
summary(Reg)
varImp(Reg)
regPredict <- predict(Reg,yelpResCol)
RMSE(regPredict, yelpResCol$stars)

imp <- as.data.frame(varImp(Reg))
imp <- data.frame(overall = imp$Overall,
                  names   = rownames(imp))
imp[order(imp$overall,decreasing = T),]
imp_order <- imp[order(imp$overall,decreasing = T),]

graphnames = imp_order$names
barplot(imp_order$overall, names.arg=graphnames, main="Attribute Importance",ylab="Overall Score", las=3, cex.names=0.7 )



########## Linear Regression (selected variables)
Reg2 <- lm(stars~attributes.RestaurantsPriceRange2+
            attributes.BusinessAcceptsCreditCards+
            attributes.NoiseLevel+
            attributes.RestaurantsReservations+
            attributes.OutdoorSeating+
            attributes.HasTV+
            attributes.WiFi+
            attributes.BusinessParking,
          data = yelpResCol)
summary(Reg2)
reg2Predict <- predict(Reg2,yelpResCol)
RMSE(reg2Predict, yelpResCol$stars)

plot(Reg2)




########## Decision Tree (selected variables)
cvtree <- train(stars ~ attributes.RestaurantsPriceRange2+
                  attributes.BusinessAcceptsCreditCards+
                  attributes.NoiseLevel+
                  attributes.RestaurantsReservations+
                  attributes.OutdoorSeating+
                  attributes.HasTV+
                  attributes.WiFi+
                  attributes.BusinessParking,
                  data = yelpResCol,  
                  method = 'ctree', 
                  trControl=trainControl(method = 'cv', number=5),
                  tuneGrid=expand.grid(mincriterion=0.95))
cvtree
plot(cvtree$finalModel, type="simple")
plot(cvtree$finalModel) #with labels on terminal node
cvtree$results

varImp(cvtree)


########## Random Forest (selected variables)
rf <- train(stars ~ attributes.RestaurantsPriceRange2+
              attributes.BusinessAcceptsCreditCards+
              attributes.NoiseLevel+
              attributes.RestaurantsReservations+
              attributes.OutdoorSeating+
              attributes.HasTV+
              attributes.WiFi+
              attributes.BusinessParking,
            data = yelpResCol,  
            method = 'rf', 
            tuneLength = 4,    
            trControl=trainControl(method = 'cv', number=5),
            linout = TRUE)
rf
summary(rf)
plot(rf)


########## Neural Network (selected variables)
nnet <- train(stars ~ attributes.RestaurantsPriceRange2+
                  attributes.BusinessAcceptsCreditCards+
                  attributes.NoiseLevel+
                  attributes.RestaurantsReservations+
                  attributes.OutdoorSeating+
                  attributes.HasTV+
                  attributes.WiFi+
                  attributes.BusinessParking,
                data = yelpResCol,  
                method = 'nnet', 
                trControl=trainControl(method = 'cv', number=5),
                tuneGrid = expand.grid(size=c(1:10), decay=seq(0.1, 1, 0.1)),
                #tuneGrid=expand.grid(mincriterion=0.95))
                linout = TRUE)
nnet
summary(nnet)
plot(nnet)




########## SVM  (limited variables)
svmreg <- svm(stars ~ attributes.RestaurantsPriceRange2+
              attributes.BusinessAcceptsCreditCards+
              attributes.NoiseLevel+
              attributes.RestaurantsReservations+
              attributes.OutdoorSeating+
              attributes.HasTV+
              attributes.WiFi+
              attributes.BusinessParking,
            data = yelpResCol)

summary(svmreg)
svmPredict <- predict(svmreg, yelpResCol)
RMSE(svmPredict, yelpResCol$stars)

R2_Score(svmPredict, yelpResCol$stars)




