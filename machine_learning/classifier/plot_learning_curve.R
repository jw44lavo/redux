library(ggplot2)
library(readr)
theme_set(
  theme_bw() +
    theme(legend.position = "top", legend.justification = "left")
    #theme(legend.position = "none")
  )

height <- 7
width  <- 7
font   <- 20
dpi    <- 300

args = commandArgs(trailingOnly=TRUE)
data <- read_csv(args[1])

train_y_min = data$train_scores_mean - data$train_scores_std
train_y_max = data$train_scores_mean + data$train_scores_std
test_y_min = data$test_scores_mean - data$test_scores_std
test_y_max = data$test_scores_mean + data$test_scores_std
data$x <- c(400, 1400, 2300, 3200, 4100)

p <- ggplot(data, aes(x)) +
geom_point(aes(y = train_scores_mean, color = "#00BFC4"), size = 3) +
geom_line(aes(y = train_scores_mean, color = "#00BFC4"), size = 1) +
geom_ribbon(aes(ymin = train_y_min, ymax = train_y_max, fill = "#00BFC4"), alpha = 0.2)+
geom_point(aes(y= test_scores_mean, color = "#F8766D"), size = 3) +
geom_line(aes(y = test_scores_mean, color = "#F8766D"), size = 1) +
geom_ribbon(aes(ymin = test_y_min, ymax = test_y_max, fill = "#F8766D"), alpha = 0.2)+

#guides(color = guide_legend(override.aes = list(size=3)))
labs(y ="Accuracy", x = "Training Samples") +
scale_colour_manual(name = "Method:",labels = c("Training", "Cross-validation"), values = c("#F8766D", "#00BFC4")) +
scale_fill_manual(name = "Method:",labels = c("Training", "Cross-validation"), values = c("#F8766D", "#00BFC4")) +
theme(text = element_text(size=font), axis.text.x = element_text(angle=0)) +
ggsave("learning_curve_ggplot.png", dpi = dpi, width = width, height = height)