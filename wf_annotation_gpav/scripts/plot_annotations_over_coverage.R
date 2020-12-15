library(ggplot2)
library(readr)
library("ggpubr")
theme_set(
  theme_bw() +
    #theme(legend.position = "top", legend.justification = "left", strip.background = element_blank(), strip.text = element_blank())
    theme(legend.position = "none")
  )

width  <- 8
height <- 14
dpi    <- 300
font   <- 20

args = commandArgs(trailingOnly=TRUE)
data <- read_csv(args[1])

data <- data[!(data$count==0),] # drop counts of 0

data <- transform(data, annotation=factor(annotation,levels=c("shovill", "plass", "direct"))) # sort by annotation to plot in wanted order shovill - plass - direct

# either drop rows with 'full' coverage or
# integrate them on next bigger tick on x-axis (1e+03)
# 'full' is parsed to NA by default by r dataframe, because it is expecting doubles
#data <- na.omit(data)  # drop
data[is.na(data)] <- 1000   # integrate

plot <- ggplot(data = data, mapping = aes(x = coverage, y = count, color = annotation))
plot <- plot + geom_point(position = "jitter", alpha = 0.8)
#plot <- plot + geom_vline(xintercept = 1.0, linetype = "dashed", color = "black", size = 0.5, alpha = 0.7)
#plot <- plot + geom_hline(yintercept = 1900, linetype = "dashed", color = "black", size = 0.5, alpha = 0.7)
#plot <- plot + geom_hline(yintercept = 950, linetype = "dashed", color = "black", size = 0.5, alpha = 0.7)
plot <- plot + scale_x_continuous(trans = "log10", breaks=c(1e-02, 1e-01, 1e+00, 1e+01, 1e+02, 1e+03), labels = c("0.01", "0.1", "1.0", "10.0", "100.0", "full"))
plot <- plot + scale_y_continuous(trans = "log10", breaks=c(1, 10, 100, 1000), labels = c("1", "10", "100", "1000"))
plot <- plot + scale_color_manual(values = c("#2ca02c", "#ff7f0e", "#1f77b4"))
plot <- plot + labs(y ="Number of unique Annotations", x = "Coverage", colour = "Annotation path:")
plot <- plot + theme(text = element_text(size=font), axis.text.x = element_text(angle=0))
plot <- plot + guides(colour = guide_legend(override.aes = list(size=5)))
ggsave("annotations_over_coverage_jitter_combined.png", dpi = dpi, width = width, height = height)

plot <- plot + facet_grid(rows = vars(annotation))
ggsave("annotations_over_coverage_jitter_seperated.png", dpi = dpi, width = width, height = height)

