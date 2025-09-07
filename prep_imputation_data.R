to_drop_per_panel = 5

# For DFCI datasets:
input_data <- readRDS("DFCI/cll_p1_ds.RDS")
out_folder <- "DFCI/cll_p1_impute/"

# or:
# input_data <- readRDS("DFCI/cll_p2_ds.RDS")
# out_folder <- "DFCI/cll_p2_impute/"

# # Read metadata for batch and condition mapping
# meta <- read.csv("DFCI/Metadata.txt", sep = "\t")
# print("Metadata loaded:")
# print(head(meta))

# # Function to map sample names to batch and condition
# map_batch_condition <- function(sample_names) {
#   batch_map <- character(length(sample_names))
#   cond_map <- character(length(sample_names))
  
#   for (i in seq_along(sample_names)) {
#     fname <- sample_names[i]
#     fname_split <- strsplit(fname, "_")[[1]]
#     s_id <- paste(fname_split[2], fname_split[3], sep = "_")
    
#     meta_row <- meta[meta$Patient.id == s_id, ]
#     if (nrow(meta_row) > 0) {
#       batch_map[i] <- as.character(meta_row$Batch[1])
#       cond_map[i] <- as.character(meta_row$Condition[1])
#     } else {
#       warning(paste("No metadata found for sample:", fname, "with Patient.id:", s_id))
#       batch_map[i] <- "Unknown"
#       cond_map[i] <- "Unknown"
#     }
#   }
  
#   return(list(batch = batch_map, condition = cond_map))
# }

# # Add batch and condition columns to the input data
# mapping <- map_batch_condition(input_data$sample)
# input_data$batch <- mapping$batch
# input_data$condition <- mapping$condition

# For Van Gassen dataset:

# input_data <- readRDS("van Gassen/vanGassen_ds.RDS")
# out_folder <- "van Gassen/impute/"

if (!dir.exists(out_folder)) {
  dir.create(out_folder, recursive = TRUE)
}

# Examine the structure of the data
print("Data dimensions:")
print(dim(input_data))
print("Column names:")
print(colnames(input_data))
print("Unique samples:")
print(unique(input_data$sample))
print("Sample counts:")
print(table(input_data$sample))
print("Batch distribution:")
print(table(input_data$batch))
print("Condition distribution:")
print(table(input_data$condition))

# Randomly split into 3 panels based on 'sample' column
set.seed(123)  # for reproducibility
unique_samples <- unique(input_data$sample)
n_samples <- length(unique_samples)

# Randomly assign samples to 3 panels
panel_assignments <- sample(rep(1:3, length.out = n_samples))
names(panel_assignments) <- unique_samples

# Create 3 panels
panel1_samples <- names(panel_assignments)[panel_assignments == 1]
panel2_samples <- names(panel_assignments)[panel_assignments == 2]
panel3_samples <- names(panel_assignments)[panel_assignments == 3]

panel1_data <- input_data[input_data$sample %in% panel1_samples, ]
panel2_data <- input_data[input_data$sample %in% panel2_samples, ]
panel3_data <- input_data[input_data$sample %in% panel3_samples, ]

# Verify that batch and condition are preserved in each panel
print("Panel 1 - Batch distribution:")
print(table(panel1_data$batch))
print("Panel 1 - Condition distribution:")
print(table(panel1_data$condition))

print("Panel 2 - Batch distribution:")
print(table(panel2_data$batch))
print("Panel 2 - Condition distribution:")
print(table(panel2_data$condition))

print("Panel 3 - Batch distribution:")
print(table(panel3_data$batch))
print("Panel 3 - Condition distribution:")
print(table(panel3_data$condition))

# save the result as 3 separate csv files (ground truth)
write.csv(panel1_data, file.path(out_folder, "panel1_gt.csv"), row.names = FALSE)
write.csv(panel2_data, file.path(out_folder, "panel2_gt.csv"), row.names = FALSE)
write.csv(panel3_data, file.path(out_folder, "panel3_gt.csv"), row.names = FALSE)

# Identify measurement columns (exclude metadata columns)
metadata_cols <- c("sample", "batch", "condition", "id")
measurement_cols <- setdiff(colnames(input_data), metadata_cols)

print("Measurement columns available:")
print(measurement_cols)
print(paste("Total measurement columns:", length(measurement_cols)))

# Randomly select different columns to drop for each panel
set.seed(456)  # seed for column selection
all_cols_to_drop <- sample(measurement_cols, to_drop_per_panel * 3)

# Assign different columns to each panel
panel1_drop_cols <- all_cols_to_drop[1:to_drop_per_panel]
panel2_drop_cols <- all_cols_to_drop[(to_drop_per_panel + 1):(to_drop_per_panel * 2)]
panel3_drop_cols <- all_cols_to_drop[(to_drop_per_panel * 2 + 1):(to_drop_per_panel * 3)]

print("Columns to drop from each panel:")
print(paste("Panel 1:", paste(panel1_drop_cols, collapse = ", ")))
print(paste("Panel 2:", paste(panel2_drop_cols, collapse = ", ")))
print(paste("Panel 3:", paste(panel3_drop_cols, collapse = ", ")))

# Create subset panels by removing the selected columns
panel1_subset <- panel1_data[, !colnames(panel1_data) %in% panel1_drop_cols]
panel2_subset <- panel2_data[, !colnames(panel2_data) %in% panel2_drop_cols]
panel3_subset <- panel3_data[, !colnames(panel3_data) %in% panel3_drop_cols]

# save the resulting subset panels as 3 csv files
write.csv(panel1_subset, file.path(out_folder, "panel1_subset.csv"), row.names = FALSE)
write.csv(panel2_subset, file.path(out_folder, "panel2_subset.csv"), row.names = FALSE)
write.csv(panel3_subset, file.path(out_folder, "panel3_subset.csv"), row.names = FALSE)

print("Subset panels saved successfully!")
print(paste("Panel 1 dimensions:", paste(dim(panel1_subset), collapse = " x ")))
print(paste("Panel 2 dimensions:", paste(dim(panel2_subset), collapse = " x ")))
print(paste("Panel 3 dimensions:", paste(dim(panel3_subset), collapse = " x ")))