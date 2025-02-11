suppressPackageStartupMessages({
    library(caret, quietly = T)
    library(tidyverse, quietly = T)
    library(glmnet)
    library(tools)
})
source('util.R')

# constants
preProc = c("nzv", "center", "scale")
ycol = "age"
cachedir = 'cache_level0'
#set high expressions which might be needed by some models like pls
# when we have high number of voxels
# see: https://stackoverflow.com/questions/32826906/how-to-solve-protection-stack-overflow-issue-in-r-studio
options(expressions = 5e5)

force = FALSE

# parse user arguments
args = commandArgs(trailingOnly=TRUE)
# csv/tsv with voxel values for a parcel 
dfile  = args[1]
# output dir
odir   = args[2]
# eg glmnet
method = args[3]
# dataset
data   = args[4]
if(length(args)>=5) {
    seed = args[5]
    stopifnot(seed>0)
} else {
    seed = sample(1:99999, 1)
}
cat('level0 with', dfile, 'method', method, 'seed', seed, '\n')

parcel = file_path_sans_ext(basename(dfile))
odir = file.path(odir, cachedir, data, method)
dir.create(odir, recursive = TRUE, showWarnings = FALSE)
ofile = file.path(odir, paste(parcel, ".RData", sep=""))
cat('output file', ofile, '\n')
if(file.exists(ofile)) {
    if(force) {
        cat('overwriting existing file', ofile, '\n')
        file.remove(ofile)
    } else {
        cat('output file exists',ofile,'\n')
        load(ofile)
        opt <- options(show.error.messages = FALSE)
        on.exit(options(opt))
        stop()
    }
}

xy = get_data_voxelwise(dfile, ycol, data)
show(dim(xy$x))

set.seed(seed)

result = list()
result$file = dfile
result$seed = seed
result$data = list()
result$data$y = xy$y
result$data$mean   = apply(xy$x,1,mean)
result$data$median = apply(xy$x,1,median)

# features from level0 models to be used for level1 models
cat('level0 CV...\n')
result = level0CV(x=xy$x, y=xy$y, k=3, result=result,
                  method=method, preProc=preProc,
                  finalModel = TRUE)

# save result
cat('saving output file',ofile,'\n')
save(result, file=ofile)

cat('done\n')


