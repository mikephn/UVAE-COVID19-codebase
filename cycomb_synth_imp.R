library(cyCombine)
library(tidyverse)

grid_size = 16

out_name <- "vg_16"

# Define input data panels:

# panel_0 <- read_csv("toy_panel0.csv")
# panel_1 <- read_csv("toy_panel1.csv")
# panel_2 <- read_csv("toy_panel2.csv")

#panel_0 <- read_csv(paste("DFCI/cll_p1_impute/panel1_subset.csv", sep = ''))
#panel_1 <- read_csv(paste("DFCI/cll_p1_impute/panel2_subset.csv", sep = ''))
#panel_2 <- read_csv(paste("DFCI/cll_p1_impute/panel3_subset.csv", sep = ''))

#panel_0 <- read_csv(paste("DFCI/cll_p2_impute/panel1_subset.csv", sep = ''))
#panel_1 <- read_csv(paste("DFCI/cll_p2_impute/panel2_subset.csv", sep = ''))
#panel_2 <- read_csv(paste("DFCI/cll_p2_impute/panel3_subset.csv", sep = ''))

panel_0 <- read_csv(paste("van Gassen/impute/panel1_subset.csv", sep = ''))
panel_1 <- read_csv(paste("van Gassen/impute/panel2_subset.csv", sep = ''))
panel_2 <- read_csv(paste("van Gassen/impute/panel3_subset.csv", sep = ''))

m_0 <- get_markers(panel_0)
m_1 <- get_markers(panel_1)
m_2 <- get_markers(panel_2)

corrected_0 <- batch_correct(panel_0, xdim = grid_size, ydim = grid_size,
                             norm_method = "scale", markers = m_0)
corrected_1 <- batch_correct(panel_1, xdim = grid_size, ydim = grid_size,
                             norm_method = "scale", markers = m_1)
corrected_2 <- batch_correct(panel_2, xdim = grid_size, ydim = grid_size,
                             norm_method = "scale", markers = m_2)

shared_01 <- intersect(m_0, m_1)
shared_02 <- intersect(m_0, m_2)
shared_12 <- intersect(m_1, m_2)

shared_markers <- intersect(m_0, m_1)
shared_markers <- intersect(shared_markers, m_2)

all_markers <- union(m_0, m_1)
all_markers <- union(all_markers, m_2)

missing_0 <- all_markers[!(all_markers %in% m_0)]
missing_1 <- all_markers[!(all_markers %in% m_1)]
missing_2 <- all_markers[!(all_markers %in% m_2)]

use_cols_01 <- c(shared_01, 'batch')
uncorrected_01 <- bind_rows(panel_0[,use_cols_01], panel_1[,use_cols_01])
corrected_01 <- batch_correct(uncorrected_01, xdim = grid_size, ydim = grid_size,
                              norm_method = "scale", markers = shared_01)

use_cols_02 <- c(shared_02, 'batch')
uncorrected_02 <- bind_rows(panel_0[,use_cols_02], panel_2[,use_cols_02])
corrected_02 <- batch_correct(uncorrected_02, xdim = grid_size, ydim = grid_size,
                              norm_method = "scale", markers = shared_02)

use_cols_12 <- c(shared_12, 'batch')
uncorrected_12 <- bind_rows(panel_1[,use_cols_12], panel_2[,use_cols_12])
corrected_12 <- batch_correct(uncorrected_12, xdim = grid_size, ydim = grid_size,
                              norm_method = "scale", markers = shared_12)


imputed_0 <- impute_across_panels(dataset1 = corrected_0, dataset2 = corrected_12,
                                  overlap_channels = shared_markers, impute_channels1 = missing_0,
                                  impute_channels2 = NULL)$dataset1

imputed_1 <- impute_across_panels(dataset1 = corrected_1, dataset2 = corrected_02,
                                  overlap_channels = shared_markers, impute_channels1 = missing_1,
                                  impute_channels2 = NULL)$dataset1

imputed_2 <- impute_across_panels(dataset1 = corrected_2, dataset2 = corrected_01,
                                  overlap_channels = shared_markers, impute_channels1 = missing_2,
                                  impute_channels2 = NULL)$dataset1

imputed_0 <- imputed_0[ , !(names(imputed_0) %in% c("id", "label"))]
imputed_1 <- imputed_1[ , !(names(imputed_1) %in% c("id", "label"))]
imputed_2 <- imputed_2[ , !(names(imputed_2) %in% c("id", "label"))]

dir.create(paste("imputation/", out_name, sep = ''), recursive = TRUE)
write_tsv(imputed_0, paste("imputation/", out_name, "/cc_imp_0.tsv", sep = ''))
write_tsv(imputed_1, paste("imputation/", out_name, "/cc_imp_1.tsv", sep = ''))
write_tsv(imputed_2, paste("imputation/", out_name, "/cc_imp_2.tsv", sep = ''))

