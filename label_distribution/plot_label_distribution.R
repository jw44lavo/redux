library(ggplot2)

theme_set(
  theme_bw() +
    #theme(legend.position = "top", legend.justification = "left", strip.background = element_blank(), strip.text = element_blank())
    theme(legend.position = "none")
  )

height <- 15
width  <- 8
font   <- 14
dpi    <- 300

df_before <- read.csv(file = 'label_distribution_before_preprocessing.csv')
df_before$preprocessing <- 'before'

df_after <- read.csv(file = 'label_distribution_after_preprocessing.csv')
df_after$preprocessing <- 'after'

data <- rbind(df_before, df_after)
data <- transform(data, preprocessing=factor(preprocessing,levels=c("before", "after"))) # sort

p <- ggplot(data = data, aes(x = reorder(Serotype, Count), y = Count, fill = preprocessing))
p <- p + geom_bar(stat = "identity")
p <- p + facet_grid(cols = vars(preprocessing))
p <- p + scale_y_continuous(breaks=c(100, 200, 300, 400, 500), labels = c("100", "200", "300", "400", "500"))
p <- p + labs(y ="Count", x = "Serotype")
p <- p + coord_flip()
p <- p + theme(text = element_text(size=font), axis.text.x = element_text(angle=0))

ggsave("label_distribution.png", p, dpi = dpi, width = width, height = height)

