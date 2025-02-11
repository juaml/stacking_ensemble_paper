#'  

library(gtools)

source('util.R')

# make sure that all the cache data is ready (see parcelwise_level0* files)
brainwise_multisite <- function(sites, method = 'glmnet', method0='glmnet', 
                                preProc = c("nzv", "center", "scale"),
                                newdatadir = NULL, newdata = 'all', newdatasample=NA,
                                outdir = file.path('..', 'models', 'multisite'),
                                cache0 = file.path('..', 'models', 'cache_level0'),
                                cache1 = file.path('..', 'models', 'cache_level1'),
                                ycol='age', testindex=NULL, 
                                dolevel2=FALSE, l0_oos_from_test=TRUE,
                                bal=NULL, balseed=NA, balnbin=5, cl=NULL) {
    # sites: array with site names (must be present in the cache)

    # collect data across sites
    sites = mixedsort(sites)
    nsites = length(sites)
    sitesstr = paste(sites, collapse='_')
    cat('sites',paste(sites),'\n')
    ofile = NULL
    models = NULL
    # if there is no output directory, create it using methods name and sites
    if(!is.null(outdir)) {
        ofile = paste('l1', method, 'l0', method0, 'train', sitesstr, sep='_')
        if(!is.null(bal)) {
            stopifnot(bal %in% c('site', 'siteage'))
            if(is.na(balseed)) balseed = sample(1:99, 1)
            ofile = paste(ofile, '_', bal, balseed, sep='')
        }
        ofile = paste(ofile, ".rda", sep="")
        ofile = file.path(outdir, ofile)
        cat('output file', ofile, '\n')
        if(file.exists(ofile)) {
            cat('model file exists',ofile,'\n')
            load(ofile) # this will load the models
        }
    }
    # create the models
    if(is.null(models)) {
        cat('creating models...\n')
        models = list()
        models$level1 = list()
        y = c()
        siteid = c()
        xmean = c()
        xage  = c()
        for(i in 1:length(sites)) {
            site = sites[i]
            cat('loading data for site', i,'/', length(sites), site, '\n')
            result = get_models_level1(site, method=method, method0=method0)
            stopifnot(!is.null(result))
            datamean = result$data$mean
            if('age' %in% names(result$data)) {
                dataage = result$data$age
            } else if('pred' %in% names(result$data)) {
                dataage = result$data$pred
            } else stop('not possible to load data for site', site)

            models$level1[[site]] = list()
            models$level1[[site]]$mean = result$models$mean
            models$level1[[site]]$age  = result$models$age

            # combined data across sites
            y = c(y, result$data$y)
            siteid = c(siteid, rep(site, length(result$data$y)))
            xmean = rbind(xmean, datamean)
            xage  = rbind(xage,  dataage)
            cat('loaded', nrow(dataage), 'subjects with', ncol(dataage), 'features\n')
        }
        stopifnot(length(y)==nrow(xmean))
        stopifnot(length(y)==nrow(xage))
        colnames(xmean) = 1:ncol(xmean)
        colnames(xage)  = 1:ncol(xage)

        if(!is.null(bal)) {
            cat('balancing sites...',bal,'\n')
            set.seed(balseed)
            ii = c()
            if(bal=='siteage') {
                ybins = cut(y, breaks=balnbin, labels=1:balnbin)
                ybins = paste(ybins, siteid, sep='_')
            } else if(bal=='site') {
                ybins = siteid
            }
            ybins = as.character(ybins)
            tbins = table(ybins)
            show(tbins)
            for(tb in names(tbins)) ii = c(ii, sample(which(ybins==tb), min(tbins)))
            y = y[ii]
            xmean = xmean[ii,]
            xage  = xage[ii,]
            cat('after balancing\n')
            show(table(ybins[ii]))
        }

        if(length(sites)>1) {
            cat('training combined level1 models...',nrow(xage),'x',ncol(xage),'\n')
            models$level1$combined = list()
            cat(method, 'mean\n')
            models$level1$combined$mean = train_model(y = y, x = xmean, method = method,
                                                preProc = preProc)
            cat(method, 'age\n')
            models$level1$combined$age  = train_model(y = y, x = xage, method = method,
                                                preProc = preProc)
        } else {
            cat('reusing single site level1 models\n')
            models$level1$combined$mean = models$level1[[site]]$mean
            models$level1$combined$age  = models$level1[[site]]$age
        }

        if(!is.null(ofile)) {
            cat('saving models',ofile,'\n')
            save(models, file=ofile)
        }
    }

    # predict
    pred = NULL
    ytest = NULL
    if(!is.null(newdatadir)) {
        cat('predicting', newdatadir, newdata, '\n')
        pred = list()
        pred$args = as.list(match.call())
        files = list.files(newdatadir, pattern='*.csv')
        files = mixedsort(files)
        cat('dealing with level0 models...\n')
        ml0 = list()
        for(site in sites) {
            cat(site, ' ')
            ml0[[site]] = get_models_level0(site, method=method0, cache=cache0, cl=cl)
            cat('\n')
        }

        ofilel0 = paste(paste('l1', method, 'l0', method0, 'train', sitesstr, 'test', newdata, sep='_'), '_pred.rda', sep='')
        ofilel0 = file.path(outdir, ofilel0)
        if(file.exists(ofilel0)) {
            cat('loading level0 predictions...',ofilel0,'\n')
            load(ofilel0)
        } else {
            # we can do predictions in parallel as this can take some time
            cat('predicting using level0 models...\n')
            if(!is.null(cl)) {
                # export variables to the cluster
                clusterExport(cl=cl, list("sites", "ycol", "newdatadir", "newdata", "testindex", 
                                          "ml0"),
                                envir=environment())
            }
            # make l0 predictions on the test data using train sites
            rescl = pblapply(files, function(f) {
                source('util.R')
                parcel = file_path_sans_ext(basename(f))
                pfile = file.path(newdatadir, f)
                x = get_data_voxelwise(pfile, ycol=ycol, data=newdata)
                if(!is.null(testindex)) {
                    x$x = x$x[testindex,]
                    x$y = x$y[testindex]
                }
                xm = apply(x$x, 1, mean)
                pp = matrix(NA, nrow=nrow(x$x), ncol=length(sites))
                colnames(pp) = sites
                for(site in sites) {
                    pp[,site] = predict(ml0[[site]][[parcel]], x$x)
                }
                return(list(xmean=xm, xpred=pp, y=x$y))
            }, cl=cl)
            #sites=sites, ycol=ycol, newdatadir=newdatadir,
            #newdata=newdata, testindex=testindex, ml0=ml0, cl=cl)

            ml0 = NULL # save memory
            pred$xmean = vector(mode = "list", length = length(files))
            pred$xage  = list()
            for(site in sites) {
                pred$xage[[site]] = vector(mode = "list", length = length(files))
            }
            # separate the results
            for(i_f in 1:length(files)) {
                y = rescl[[i_f]]$y
                pred$xmean[[i_f]] = rescl[[i_f]]$xmean
                pred$level0[[i_f]] = rescl[[i_f]]$xpred
                for(site in sites) {
                    pred$xage[[site]][i_f] = rescl[[i_f]]$xpred[,site]
                }
            }
            rescl = NULL # save memory

            pred$xmean = do.call(cbind, pred$xmean)
            colnames(pred$xmean) = 1:ncol(pred$xmean)
            for(site in sites) {
                pred$xage[[site]]  = do.call(cbind, pred$xage[[site]])
                colnames(pred$xage[[site]])  = 1:ncol(pred$xage[[site]])
                stopifnot(all(dim(pred$xmean)==dim(pred$xage[[site]])))
            }
            show(dim(pred$xmean))
            pred$y = y

            cat('predicting using level1 models... ')
            pred$level1 = list()
            cat('using OOS data from train site-> test site\n')
            aa = matrix(NA, ncol=length(sites), nrow=length(pred$y))
            colnames(aa) = sites
            mm = aa
            for(site in sites) {
                cat(site, ' ')
                aa[,site] = predict(models$level1[[site]]$age, pred$xage[[site]])
                mm[,site] = predict(models$level1[[site]]$mean, pred$xmean)
            }

            if(l0_oos_from_test) {
                cat('using OOS data from within test site\n')
                pred$age_test_oos = get_pred_level0(newdata, method0, cl)
                pred$level1$newdatal0 = matrix(NA, ncol=length(sites), nrow=length(pred$y))
                colnames(pred$level1$newdatal0) = sites
                for(site in sites) {
                    pred$level1$newdatal0[,site] = predict(models$level1[[site]]$age, pred$age_test_oos)
                }
                pred$level1$newdatal0_sitemean = apply(pred$level1$newdatal0, 1, mean)
            }

            cat('\n')

            pred$level1$mean_sites = mm
            pred$level1$mean_sitemean = apply(mm, 1, mean)

            pred$level1$age_sites = aa
            pred$level1$age_sitemean = apply(aa, 1, mean)

            save(pred, file=ofilel0)
        }

        cat('mean_sitemean\n')
        pred$level1$mean_sitemean = predict_print(model=NULL, testdata=NULL,
                                                 pred= pred$level1$mean_sitemean, 
                                                 ytest=pred$y)

        cat('age_sitemean\n')
        pred$level1$age_sitemean = predict_print(model=NULL, testdata=NULL,
                                                 pred= pred$level1$age_sitemean, 
                                                 ytest=pred$y)

        if(l0_oos_from_test) {
            cat('l0_oos_from_test_sitemean\n')
            pred$level1$newdatal0_sitemean = predict_print(model=NULL, testdata=NULL,
                                                 pred= newdatal0_sitemean,
                                                 ytest=pred$y)
        }

        if(length(sites)>1) {
            cat('predicting using pooled level1 models...\n')
            cat('mean\n')
            pred$level1$mean_combined = predict_print(models$level1$combined$mean, pred$xmean, pred$y)
            cat('age\n')
            pred$level1$age_combined  = predict_print(models$level1$combined$age, pred$xage, pred$y)
            if(l0_oos_from_test) {
                cat('l0_oos_from_test\n')
                pred$level1$newdatal0_combined  = predict_print(models$level1$combined$age, pred$age_test_oos, pred$y)
            }
        }
    }

    if(nsites>1 && dolevel2) {
        if(!('level2' %in% names(models))) {
            cat('training level2 models...', ofile, '\n')
            models$level2 = list()
            models$data = list()
            models$data$oos = get_oos_level1(sites, cache=cache1, l0_oos_from_test=l0_oos_from_test)
            cat('level2 mean\n')
            models$level2$mean = train(y=models$data$oos$y,
                                 x=models$data$oos$mean,
                                 method = method, preProc = preProc)
            cat('level2 age\n')
            models$level2$age  = train(y=models$data$oos$y,
                                 x=models$data$oos$age,
                                 method = method, preProc = preProc)
            save(models, file=ofile)
        }

        if(!is.null(newdatadir)) {
            pred$level2 = list()
            cat('predicting using level2 models\n')
            cat('mean\n')
            pred$level2$mean = predict_print(models$level2$mean, pred$level1$mean_sites, pred$y)
            cat('age\n')
            pred$level2$age = predict_print(models$level2$age, pred$level1$age_sites, pred$y)
        }
    }

    return(list(models=models, pred=pred))
}
