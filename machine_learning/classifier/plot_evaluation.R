library(ggplot2)
library(readr)
library(reshape2)
theme_set(
  theme_bw() +
    theme(legend.position = "top", legend.justification = "left")
    #theme(legend.position = "none")
  )

height <- 8
width  <- 10
font   <- 20
dpi    <- 300

args = commandArgs(trailingOnly=TRUE)
data <- read_csv(args[1])
data <- transform(data, coverage = as.numeric(coverage))
data[is.na(data)] <- 1000
data$min_appearance[(data$min_appearance)=="0"] <- "0 (82 classes)"
data$min_appearance[(data$min_appearance)=="50"] <- "50 (31 classes)"
data$min_appearance[(data$min_appearance)=="100"] <- "100 (17 classes)"
data$min_appearance[(data$min_appearance)=="200"] <- "200 (8 classes)"
data <- transform(data, min_appearance=factor(min_appearance,levels=c("0 (82 classes)", "50 (31 classes)", "100 (17 classes)", "200 (8 classes)"))) # sort

plot <- ggplot(data = data, mapping = aes(x = coverage, y = accuracy)) +
geom_point(aes(color = factor(model)), size = 3) +
geom_line(aes(color = model), alpha = 0.5, size = 1) +
scale_x_continuous(trans = "log10", breaks=c(1e-02, 1e-01, 1e+00, 1e+01, 1e+02, 1e+03), labels = c("0.01", "0.1", "1.0", "10.0", "100.0", "full")) +
ylim(0.0, 1.0) +
labs(y ="Accuracy", x = "Coverage", color = "Classifier") +
theme(text = element_text(size=font), axis.text.x = element_text(angle=0)) +
guides(color = guide_legend(override.aes = list(size=3))) +
#facet_grid(cols = vars(min_appearance))
scale_color_manual(labels = c("Gradient Boosting", "Most Frequent", "Neural Network"), values = c("#F8766D", "#00BFC4", "#7CAE00")) +
facet_wrap(~min_appearance, nrow = 2, labeller = label_both)

ggsave("mlearning_evaluation.png", plot, dpi = dpi, width = width, height = height)