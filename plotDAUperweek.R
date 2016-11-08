library(ggplot2)
library(scales)
library(plyr)

df = read.csv("DAU-Sep_Oct.csv", header=F, stringsAsFactors = F)
names(df) <- c("Day", "Users")
# remove last day (missing data)
df <- df[-nrow(df),]
df$Day <- as.Date(df$Day)
df$DayOfWeek <- weekdays(df$Day)
df$Weekdayend <- ifelse(df$DayOfWeek=='Saturday' | df$DayOfWeek=='Sunday', 'Weekend', 'Weekday')
weekday_names <- c('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
df$DayOfWeek <- factor(df$DayOfWeek, weekday_names, ordered = T)
df$WeekNumber <- as.numeric( format(df$Day+3, "%U"))
df$Month <- factor(months(df$Day), levels=month.name, ordered = T)

mean_per_month_per_day <- ddply(df, .(Month,DayOfWeek), summarize, avgvalue=mean(Users))
mean_values_per_weekday <- ddply(df, .(DayOfWeek), summarize, avgvalue=mean(Users))
mean_values_per_month <- ddply(df, .(Month), summarize, avgvalue=mean(Users))

ggplot(data = df, aes(x = WeekNumber, y=Users, group=Month, colour=Month)) +
    geom_line() +
    geom_point() +
    geom_hline(aes(yintercept=avgvalue), data=mean_values_per_weekday) +
    facet_grid(.~DayOfWeek) +
    scale_y_continuous(labels = comma)

ggplot(data = df, aes(x = WeekNumber, y=Users, group=Month, colour=Month)) +
    geom_line() +
    geom_point() +
    geom_hline(aes(yintercept=avgvalue, colour=Month), data=mean_values_per_month) +
    facet_grid(.~DayOfWeek) +
    scale_y_continuous(labels = comma)

ggplot(data = df, aes(x = WeekNumber, y=Users, group=Month, colour=Month)) +
    geom_line() +
    geom_point() +
    geom_hline(aes(yintercept=avgvalue, colour=Month), data=mean_per_month_per_day) +
    facet_grid(.~DayOfWeek) +
    scale_y_continuous(labels = comma)