library(ggplot2)
library(readr)

width  <- 12
height <- 12
dpi    <- 300
font   <- 20

args = commandArgs(trailingOnly=TRUE)
data <- read_csv(args[1])

library(ggplot2)
ggplot(data) + geom_point(aes(x=x, y=y, color=label)) +
guides(color = guide_legend(nrow = 8, byrow = FALSE, override.aes = list(size = 5))) +
labs(color = "Serotype") + 
theme(text = element_text(size=font), axis.text.x = element_text(angle=0)) +
theme(legend.position="bottom")

ggsave("tsne_embedding.png", dpi = dpi, width = width, height = height)

