library(gtools)
source('brainwise_multisite.R')

# parse user arguments
args = commandArgs(trailingOnly=TRUE)
# train datasets
sites  = args[1]
# dir with test dataset
newdatadir = args[2]
# test dataset name
newdata = args[3]

if(length(args) > 3) {
    l0_oos_from_test= args[4]
} else {
    l0_oos_from_test = TRUE
}

# convert to bool
l0_oos_from_test = as.logical(l0_oos_from_test)



cat('use models trained with ', sites, 'to predict',
    newdata, 'from folder', newdatadir, '\n')

tr_sites = strsplit(sites, split=",")

res = brainwise_multisite(sites=tr_sites[[1]], newdatadir=newdatadir, newdata=newdata, l0_oos_from_test=l0_oos_from_test)
