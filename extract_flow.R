# Title     : Extract FCS samples from multiple WSP workspaces to a single HDF5 file.
# Objective : Load panel and label mappings. Add automatic debris and doublet labels where required. Save subsamples to a single HDF file, optionally with extra labeled data.
# Created by: Mike Phuycharoen
# Created on: 8-MAR-2021

library(CytoML)
library(openCyto)
library(ggcyto)
library(flowCore)
library(grid)
library(gridExtra)
library(rhdf5)

sourceDir <- './Flow cytometry data/valid/' # where wsp files are
outFolder <- './Flow cytometry data/out/' # where to save dataset and plots
dir.create(outFolder)
includeRawFcs <- FALSE # include files from outside workspaces

# find workspaces
allWspFiles <- list.files(sourceDir, pattern = "\\.wsp$", recursive = TRUE)

chemokineFiles <- allWspFiles[grepl("chemo", basename(allWspFiles), ignore.case = TRUE)]
pbmcFiles <- allWspFiles[grepl("pbmc", basename(allWspFiles), ignore.case = TRUE)]
lineageFiles <- setdiff(allWspFiles, c(chemokineFiles, pbmcFiles))
lineageFiles <- lineageFiles[!startsWith(lineageFiles, 'B) Chemokine')]

# which workspaces to use
wspFiles <- lineageFiles

# create dataset file
dsFilePath <- file.path(outFolder, "lineage.h5")
dsFile <- h5createFile(dsFilePath)

# minimum number of labeled cells in a file in order to upsample a class
minCellsToUpsample <- 100

isToyData <- FALSE
# load mapping of nodes to labels
nodeMappingPath <- './Flow cytometry data/nodes.tsv'
if (isToyData) {
  nodeMappingPath <- './Flow cytometry data/nodes_toy.tsv'
}
unkLabel <- '-'
# load table with paths in column 1 and labels in consecutive columns
node_mapping <- read.table(nodeMappingPath, sep='\t', stringsAsFactors = FALSE, header = TRUE)
# establish enumeration levels
label_types <- NULL
label_levels <- list()
for (cn in 2:ncol(node_mapping)) {
  # replace blanks with unknown token -
  node_mapping[node_mapping[, cn] == '', cn] <- unkLabel
  all_classes <- levels(as.factor(node_mapping[, cn]))
  if (!(unkLabel %in% all_classes)) {
    all_classes <- c(unkLabel, all_classes)
  }
  series_name <- colnames(node_mapping)[cn]
  label_types <- c(label_types, series_name)
  label_levels[[series_name]] <- all_classes
}

# attribute for storing channels in hdf
chAttrName <- 'captions'

extractSamples <- function (gs, filename, panelGroup, nRandom=10000, nClass=1000, subsample=100000, gateDoublets=TRUE, gateDebris=FALSE, applyLogicle=TRUE, makePlot=FALSE) {
    readLabeling <- function (gs, series) {
      # create label vectors
      root_mask <- gh_pop_get_indices(gs[[1]], 'root')
      cell_labels <- as.factor(rep(unkLabel, length(root_mask)))
      levels(cell_labels) <- label_levels[[series]]
      for (ln in seq_len(nrow(node_mapping))) {
        node <- node_mapping[ln, 1]
        label <- node_mapping[ln, series]
        if (label != unkLabel) {
          try({
            node_mask <- gh_pop_get_indices(gs[[1]], node)
            cell_labels[node_mask] <- label
          }, silent=TRUE)
        }
      }
      return(cell_labels)
    }

    hdfGroupExists <- function (dsFilePath, group) {
      fid <- H5Fopen(dsFilePath)
      groupExists <- H5Lexists(fid, group)
      H5Fclose(fid)
      return(groupExists)
    }

    hdfWrite <- function (X, filePath, groupPath, captions=NULL) {
      if (!is.null(captions)) {
        attr(X, chAttrName) <- captions
        h5write(X, file = filePath, name=groupPath, write.attributes=TRUE)
      } else {
        h5write(X, file = filePath, name=groupPath)
      }
    }

    # check if panel entry exists
    if (!hdfGroupExists(dsFilePath, panelGroup)) {
      h5createGroup(dsFilePath, group = panelGroup)
    }
    # check if data was already added, skip file if true
    fileGroup <- paste(panelGroup, filename, sep = '/')
    dataGroup <- paste(fileGroup, 'random', 'X', sep='/')
    if (hdfGroupExists(dsFilePath, dataGroup)) {
      return(NULL)
    }
    h5createGroup(dsFilePath, group = fileGroup)

    fr <- gh_pop_get_data(gs[[1]], 'root')
    if (applyLogicle) {
      fr <- flowCore::transform(fr, flowCore::transformList(colnames(fr), flowCore::logicleTransform()))
    }
    expr <- exprs(fr)

    # correct channel names
    channels <- colnames(gs)
    markers <- NULL
    for (ci in seq_along(channels)) {
      cn <- channels[ci]
      p <- getChannelMarker(fr, cn)
      if (is.na(p[['desc']]) == FALSE) {
        cn <- p[['desc']]
      }
      if (cn == '') {
        cn <- 'NA'
      }
      markers <- c(markers, cn)
    }
    colnames(expr) <- markers

    # read all manual labels
    labels <- list()
    for (series in label_types) {
      labels[[series]] <- readLabeling(gs, series)
    }

    # use first column labels for upsampling
    cell_labels <- labels[[label_types[1]]]
    # save extra labeled samples separately to avoid altering proportions in random data
    if (nClass > 0) {
      extraInds <- NULL
      foundClasses <- 0
      for (c in levels(cell_labels)) {
        if (c != unkLabel) {
          c_inds <- which(cell_labels == c)
          if (length(c_inds) >= minCellsToUpsample) {
            toSample <- sample(length(c_inds), min(length(c_inds), nClass))
            extraInds <- c(extraInds, c_inds[toSample])
            foundClasses <- foundClasses + 1
          }
        }
      }
      if (isToyData) {
        if (foundClasses < 7) {
          return(NULL)
        }
      }
      if (foundClasses > 0) {
        X <- expr[extraInds,]
        h5createGroup(dsFilePath, group = paste(fileGroup, 'extra', sep = '/'))
        hdfWrite(t(X), dsFilePath, paste(fileGroup, 'extra', 'X', sep = '/'), captions = markers)
        for (series in label_types) {
          Y <- labels[[series]][extraInds]
          hdfWrite(as.numeric(Y)-1, dsFilePath, paste(fileGroup, 'extra', series, sep = '/'), captions = levels(Y))
        }
      }
    }

    if (subsample) {
      # subsample for gating
      rs <- sample(nrow(expr), min(nrow(expr), subsample))
      expr <- expr[rs,]
      fr <- flowFrame(expr)
      for (series in label_types) {
        labels[[series]] <- labels[[series]][rs]
      }
    }

    # plot placeholders
    p0 <- NULL
    p1 <- NULL
    p2 <- NULL
    p3 <- NULL

    if (gateDoublets) {
      g <- GatingSet(flowSet(fr))
      t <- gs_add_gating_method(g, alias = "Doublets", pop = "-", parent = "root",
                               dims = "FSC-A,FSC-H", gating_method = "singletGate")
      is_doublet <- gh_pop_get_indices(g, 'Doublets')
      if (makePlot) {
        p3 <- ggcyto(g, aes(x = 'FSC-A', y = 'FSC-H')) + geom_hex(bins = 60) + geom_gate('Doublets')
      }
    }

    if (gateDebris) {
      # normalise a copy for this gating to get more consistent results across many files
      fr_deb <- flowFrame(expr)
      for (chn in colnames(fr_deb)) {
        vals <- exprs(fr_deb)[, chn]
        ch_mean <- mean(vals)
        ch_sd <- sd(vals)
        vals <- vals - ch_mean
        vals <- vals / ch_sd
        exprs(fr_deb)[, chn] <- vals
      }
      g_deb <- GatingSet(flowSet(fr_deb))
      t <- gs_add_gating_method(g_deb, alias = "LowSSCFSC", pop = "+",parent = "root",
                               dims = "FSC-A,SSC-A", gating_method =
                                 "flowClust",preprocessing_method = "prior_flowClust",
                               preprocessing_args="K=4",gating_args =
                                 "quantile=0.9, K=4, transitional=T, translation = -0.9, target=c(-3,-3), transitional_angle=pi*1.25, plot=FALSE")

      t <- gs_add_gating_method(g_deb, alias = "LowSSCCD45", pop = "+",parent = "root",
                               dims = "CD45,SSC-A", gating_method =
                                 "flowClust",preprocessing_method = "prior_flowClust",
                               preprocessing_args="K=4",gating_args =
                                 "quantile=0.9, K=4, transitional=T, translation = -1.0, target=c(-3,-3), transitional_angle=pi*1.25, plot=FALSE")

      t <- gs_add_gating_method(g_deb, alias = "Debris", gating_method = "boolGate",
                               parent = "root", gating_args = "LowSSCFSC&LowSSCCD45")

      is_debris <- gh_pop_get_indices(g_deb, 'Debris')

      if (makePlot) {
        p0 <- ggcyto(g_deb, aes(x = 'FSC-A', y = 'SSC-A')) + geom_hex(bins = 60) + geom_gate('LowSSCFSC') + geom_stats()
        p1 <- ggcyto(g_deb, aes(x = 'CD45', y = 'SSC-A')) + geom_hex(bins = 60) + geom_gate('LowSSCCD45') + geom_stats()
        p2 <- ggcyto(g_deb, aes(x = 'FSC-A', y = 'SSC-A')) + geom_hex(bins = 60) + geom_gate('Debris') + geom_stats()
      }
    }

    p <- NULL
    if (makePlot && !(is.null(p0) || is.null(p1) || is.null(p2) || is.null(p3))) {
      # generate plot
      p <- gridExtra::arrangeGrob(as.ggplot(p0), as.ggplot(p1), as.ggplot(p2), as.ggplot(p3), ncol=2, nrow = 2, top=filename)
    }

    # save random sample
    rs <- sample(nrow(expr), min(nrow(expr), nRandom))
    X <- expr[rs,]
    h5createGroup(dsFilePath, group = paste(fileGroup, 'random', sep = '/'))
    # save data
    hdfWrite(t(X), dsFilePath, dataGroup, captions = markers)
    # save labels
    for (series in label_types) {
      labels[[series]] <- Y <- labels[[series]][rs]
      hdfWrite(as.numeric(Y)-1, dsFilePath, paste(fileGroup, 'random', series, sep = '/'), captions=levels(Y))
    }
    if (gateDoublets) {
      Ydb <- is_doublet[rs]
      hdfWrite(Ydb, dsFilePath, paste(fileGroup, 'random', 'doublets', sep = '/'), captions=c('Singlet', 'Doublet'))
    }
    if (gateDebris) {
      Ydeb <- is_debris[rs]
      hdfWrite(Ydeb, dsFilePath, paste(fileGroup, 'random', 'debris', sep = '/'), captions=c('Live', 'Debris'))
    }

    return(list(X = expr, ctype = labels[[label_types[1]]], plot=p))
}

wspSamples <- NULL # record which samples were saved from workspaces

for (wsfile in wspFiles) {
  path <- file.path(sourceDir, wsfile)
  ws <- open_flowjo_xml(path)
  samples <- fj_ws_get_samples(ws)
  valid <- !startsWith(samples[[2]], 'Compensation') # filter valid samples from workspace

  sampleNames <- samples[[2]][valid]
  sampleIds <- samples[[1]][valid]
  if (length(sampleNames) == 0) {
    next
  }

  print(path)
  print(sampleNames)
  # accumulate plots from a workspace
  plots <- list()

  for (n in seq_along(sampleIds)) {
    sample_id <- sampleIds[n]
    sample_name <- sampleNames[n]
    if (isToyData) {
      if (grepl('HC', sample_name) == FALSE) {
        next
      }
    }
    if (sample_name %in% wspSamples) {
      print('Duplicate:')
      print(sample_name)
      next
    }
    print(sample_name)
    fcsPath <- file.path(sourceDir, dirname(wsfile), sample_name)
    if (file.exists(fcsPath) == FALSE) {
      print('File missing:')
      print(fcsPath)
      next
    }

    tryCatch({
      gs <- flowjo_to_gatingset(ws, name=1, subset=sample_name, isNcdf = TRUE)
      res <- extractSamples(gs, sample_name, panelGroup = basename(wsfile))
      if (!is.null(res)) {
        wspSamples <- c(wspSamples, sample_name)
        p <- res[['plot']]
        if (!is.null(p)) {
          plots[[n]] <- p
        }
        print(table(res['ctype']))
      }
    }, error = function(e) {print(e)})
  }

  if (length(plots) > 0) {
    # save all plots in one pdf per workspace
    pdf(file.path(outFolder, sub('.wsp', '.pdf', basename(wsfile))))
    for (i in seq_along(plots)) {
      grid::grid.newpage()
      grid::grid.draw(plots[[i]])
    }
    dev.off()
  }
}

if (includeRawFcs) {
  # add FCS files outside workspaces
  fcsPaths <- list.files(sourceDir, pattern = "\\.fcs", recursive = TRUE) # find all FCS files
  fcsFiles <- basename(fcsPaths)
  for (n in seq_along(fcsFiles)) {
    filename <- fcsFiles[n]
    if (filename %in% wspSamples) {
      next
    }
    if (startsWith(filename, 'Compensation')) {
      next
    }
    tryCatch({
      path <- file.path(sourceDir, fcsPaths[fcsFiles == filename])
      fcs <- read.flowSet(path)
      gs <- GatingSet(fcs)
      comp_fcs <- spillover(fcs[[1]])[["SPILL"]]
      comp_fcs <- compensation(comp_fcs)
      gs <- compensate(gs, comp_fcs)
      path_comps = strsplit(path, '/')[[1]]
      panelName <- path_comps[[length(path_comps)-1]]
      print(filename)
      res <- extractSamples(gs, filename, panelGroup = panelName, makePlot=FALSE, gateDoublets = TRUE)
    }, error = function(e) {print(e)})
  }
}

h5closeAll()
