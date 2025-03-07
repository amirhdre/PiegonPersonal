# install.packages("immunarch")
library(immunarch)
library(dplyr)
library(ggplot2)

# Load the data
immdata <- repLoad("C:/Users/administer/Desktop/PCM__EHR/pigeon/data/TCR Raw Data/Pt6")
# imm_gu <- geneUsage(immdata$data, "hs.trbv")
imm_gu <- geneUsage(immdata$data, "hs.trbv", .norm = T)

imm_gu_js <- geneUsageAnalysis(imm_gu, .method = "js", .verbose = F)
imm_gu_cor <- geneUsageAnalysis(imm_gu, .method = "cor", .verbose = F)

p1 <- vis(imm_gu_js, .title = "Gene usage JS-divergence", .leg.title = "JS", .text.size = 5)
p2 <- vis(imm_gu_cor, .title = "Gene usage correlation", .leg.title = "Cor", .text.size = 5)

print(p1 + p2)
imm_gu_js[is.na(imm_gu_js)] <- 0

p3 <- vis(geneUsageAnalysis(imm_gu, "cosine+hclust", .verbose = F))
print(p3)
# imm_cl_pca <- geneUsageAnalysis(imm_gu, "js+pca+kmeans", .verbose = F)
# imm_cl_mds <- geneUsageAnalysis(imm_gu, "js+mds+kmeans", .verbose = F)
# imm_cl_tsne <- geneUsageAnalysis(imm_gu, "js+tsne+kmeans", .perp = .5, .verbose = F)
# p1 <- vis(imm_cl_pca, .plot = "clust")
# p2 <- vis(imm_cl_mds, .plot = "clust")
# p3 <- vis(imm_cl_tsne, .plot = "clust")
# print(p1 + p2 + p3)

imm_cl_pca2 <- geneUsageAnalysis(imm_gu, "js+pca+kmeans", .k = 2, .verbose = F)
p4 <- vis(imm_cl_pca2)
print(p4)
# 保存图像并设置大小
# ggsave("combined_plot.png", plot = p1+p2+p3, width = 10, height = 8)