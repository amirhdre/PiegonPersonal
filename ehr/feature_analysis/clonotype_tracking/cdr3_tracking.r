# install.packages("immunarch")
library(immunarch)
library(dplyr)
library(ggplot2)

# Load the data
for (i in 6:21) {
  immdata <- repLoad(paste0("D:/desktop/PCM__EHR/pigeon/data/TCR Raw Data/Pt", i))
  tc1 <- trackClonotypes(immdata$data, list(1, 10), .col = "aa")
  p1 <- vis(tc1, .plot = "smooth") + scale_fill_brewer(palette = "RdBu")
  ggsave(paste0("clonotype_tracking_aa_Pt", i, ".png"), plot = p1, width = 16, height = 12)
}
# immdata <- repLoad('D:/desktop/PCM__EHR/pigeon/data/TCR Raw Data/Pt1')

# tc1 <- trackClonotypes(immdata$data, list(1, 10), .col = "nt")

# p1 <- vis(tc1, .plot = "smooth") + scale_fill_brewer(palette = "Spectral")

# print(p1)
# ggsave("clonotype_tracking_Pt1.png", plot = p1, width = 16, height = 12)