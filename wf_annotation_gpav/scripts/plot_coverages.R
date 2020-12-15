library(ggplot2)
library(readr)

width  <- 8
height <- 4
dpi    <- 300
font   <- 16

args = commandArgs(trailingOnly=TRUE)
data <- read_csv(args[1])
data <- data[!(data$coverage > 1000),] # drop coverages bigger than 1000, cause it makes an ugly histogram

plot <- ggplot(data = data, aes(x=coverage))
plot <- plot + geom_histogram(fill = "#1f77b4", binwidth = 10)
plot <- plot + geom_vline(aes(xintercept=median(data$coverage), color="median"), linetype="dashed", size=0.6)
plot <- plot + geom_vline(aes(xintercept=mean(data$coverage), color="mean"), linetype="dashed", size=0.6)
plot <- plot + scale_color_manual(name = "statistics", values = c(median = "red", mean = "blue"))
plot <- plot + scale_x_continuous(breaks=c(0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000))
plot <- plot + labs(y ="Frequency", x = "Coverages")
plot <- plot + theme(text = element_text(size=font), axis.text.x = element_text(angle=90, hjust=1, vjust=0.5), legend.position = "top", legend.justification = "left")
ggsave("coverages_histogram.png", dpi = dpi, width = width, height = height)
