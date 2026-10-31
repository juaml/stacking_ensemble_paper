suppressPackageStartupMessages({
    library(caret, quietly = T)
    library(tidyverse, quietly = T)
    library(tools)
    library(gtools)
    library(txtplot)
    library(CovBat)
})

source('util.R')

# constants
preProc = c("nzv", "center", "scale")

cachedir = file.path('..', 'models', 'cache_level0')

args = commandArgs(trailingOnly=TRUE)
stopifnot(length(args)>=2)

# which algorithm to use for meta-model
method = args[1]
if(method=="" || method=="NA") method = 'glmnet'

site  = args[2] # this looks like DATA1_0levelalgo_SEED-FOLD

method0 = method
if(length(args)>2) method0 = args[3]

outdir = file.path('..', 'models', 'cache_level1')
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
outfile = file.path(outdir, paste(method,'_',method0,'_',site,'.rda', sep=''))
cat('outfile:', outfile, '\n')

cachedir = file.path(cachedir, site, method0)
cat('loading level0 data from', cachedir, '\n')
data = get_data_parcelwise(site, method0)

models = list()
cat('training level1 models...',nrow(data$pred),'x',ncol(data$pred),'\n')
cat(site, 'mean\n')
models$mean = train_model(y = data$y,
                    x = data$mean, method = method,
                    preProc = preProc)
show(models$mean)

cat(site, 'age\n')
models$age  = train_model(y = data$y,
                    x = data$pred, method = method,
                    preProc = preProc)
show(models$age)


result = list()
result$data = data
result$models = models
save(result, file = outfile)
