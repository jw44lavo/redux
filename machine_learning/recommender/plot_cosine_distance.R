library(ggplot2)
library(readr)
library(reshape2)
theme_set(
  theme_bw() +
    #theme(legend.position = "top", legend.justification = "left", strip.background = element_blank(), strip.text = element_blank())
    theme(legend.position = "none")
  )

height <- 8
width  <- 12
font   <- 20
dpi    <- 300

# cosine distance 
args = commandArgs(trailingOnly=TRUE)
data <- read_csv(args[1])
data <- transform(data, coverage = as.numeric(coverage))
data[is.na(data)] <- 1000

p <- ggplot(data, aes(x=coverage, y=cos_distance, group = coverage)) + 
geom_boxplot() +
scale_x_continuous(trans = "log10", breaks=c(1e-02, 1e-01, 1e+00, 1e+01, 1e+02, 1e+03), labels = c("0.01", "0.1", "1.0", "10.0", "100.0", "full")) +
labs(y ="Cosine Distance", x = "Coverage") +
theme(text = element_text(size=font), axis.text.x = element_text(angle=0))
ggsave("cosine_distance_recommender.png", p, dpi = dpi, width = width, height = height)