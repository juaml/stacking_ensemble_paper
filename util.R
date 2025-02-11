suppressPackageStartupMessages({
    library(tools)
    library(caret)
    # library(CovBat)
    library(gtools)
    library(pbapply)
    library(transport)
})

# source('krrCor.R')

separate_seedfold <- function(seedfold, k=NULL) {
    seedfold = strsplit(seedfold, '-')[[1]]
    stopifnot(length(seedfold) == 2)
    seed = as.integer(seedfold[1])
    fold = as.integer(seedfold[2])
    stopifnot(fold > 0)
    if(!is.null(k)) stopifnot(fold <= k) # must be sane
    return(list(seed=seed, fold=fold))
}

predict_print <- function(model, testdata, ytest, pred=NULL) {
    if(is.null(pred)) pred = predict(model, testdata)
    if(!is.null(ytest)) {
        stopifnot(nrow(testdata)==length(ytest))
        MAE = mean(abs(ytest - pred))
        cat('MAE', MAE,'\n')
        corr = cor(ytest, pred)
        cat('cor', corr,'\n')
        r2 = 1 - (sum((ytest - pred)^2) / sum((ytest - mean(ytest))^2))
        cat('r2', r2,'\n')
        bias = cor(ytest, pred-ytest)
        cat('bias', bias,'\n')
        w1d = wasserstein1d(ytest, pred)
        cat('w1d', w1d,'\n')
    } else {
        MAE  = NULL
        corr = NULL
        r2 = NULL
        bias = NULL
        w1d  = NULL
    }
    return(list(pred=pred, MAE=MAE, corr=corr, r2=r2, bias=bias, w1d=w1d, y=ytest))
}

train_model <- function(x, y, method='glmnet', allowParallel=FALSE,
                        preProc=c("nzv", "center", "scale"), ...) {
    cat('training', method, 'nrow =', nrow(x), 'ncol =', ncol(x),'\n')

    # allowParallel=FALSE is needed for 'krrCor' otherwise get `Something is wrong`
    # see: https://stackoverflow.com/questions/33308648/caret-train-method-not-working-something-is-wrong-for-all-accuracy-results-for
    ctrl <- trainControl(method = "cv", number = 10, allowParallel=allowParallel)
    if(method=='krrCor') {
        # not sure if preProc here should be used or not
        # very likely not as it will mess up the correlation
        if(!is.null(preProc) && length(preProc)>0) {
            preProc = setdiff(preProc, c("center", "scale"))
        }
        model = train(y = y, x = x, method = krrCor,
                      trControl = ctrl, preProc=preProc, ...)
    } else if(method=='xgbTree') {
        # to fix the warnings
        model = train(y = y, x = x, method = 'glmnet', preProc = preProc,
                      objective = "reg:squarederror", trControl = ctrl, ...)
    } else if(method=='pca') {
        model = prcomp(x, rank=1, retx=TRUE)
    } else if(method=='glmnetnonneg') {
        model = train(y = y, x = x, method = 'glmnet', lower.limits = 0, 
                      preProc = preProc, trControl = ctrl, ...)
    } else if(method=='glmnetnonpos') {
        model = train(y = y, x = x, method = 'glmnet', upper.limits = 0, 
                      preProc = preProc, trControl = ctrl, ...)
    } else if(method=='rf') {
        model = train(y = y, x = x, method = 'rf', upper.limits = 0, 
                      preProc = preProc, trControl = ctrl, tuneLength = 1, ...)
    } else if(method=='ridge') {
        # not using caret's inbuilt "ridge" which uses elasticnet and seems to be slow
        # when alpha=0 then lambda.max=Inf (see glmnet documentation)
        lambdas <- c(seq(0, 1, length=10), seq(2, 200, length=20))
        model = train(y = y, x = x, method = 'glmnet', 
                      preProc = preProc, trControl = ctrl, 
                      tuneGrid=expand.grid(alpha=0, lambda=lambdas), ...)
    } else if(method=='pls5') {
        model = train(y = y, x = x, method = 'pls',
                      preProc = preProc, trControl = ctrl,
                      tuneGrid=expand.grid(ncomp=5))
    } else {
        model = train(y = y, x = x, method = method,
                      preProc = preProc, trControl = ctrl, ...)
    }

    model
}

get_data_80_20 <- function(setup, setup2=NULL, datadir=NULL, files=NULL,
                    finalModel=FALSE, ycol='age', cacheagepred=TRUE,
                    cache=file.path('..', 'models', 'cache_level0_pred')) {
    # when datadir is provided then setup is used to predict setup2 test data
    setupin = setup
    setup2in = setup2

    setup = strsplit(setup, '_')[[1]]
    stopifnot(length(setup)==3)
    ddir  = file.path('..', 'models', '80_20', setup[1], setup[2], setup[3])
    if(is.null(files)) files = list.files(ddir, pattern='*.RData')

    ddir2 = NULL
    if(!is.null(setup2)) {
        setup2 = strsplit(setup2, '_')[[1]]
        stopifnot(length(setup2)==3)
        ddir2  = file.path('..', 'models', '80_20', setup2[1], setup2[2], setup2[3])
        files2 = list.files(ddir2, pattern='*.RData')
        files = intersect(files, files2)
    }
    files = mixedsort(files) # read them in same order

    Xtrain = list()
    Xtrain$mean = list()
    Xtrain$age  = list()

    Xtest = list()
    Xtest$mean = list()
    Xtest$age = list()
    Xtest$agepred = list()
    cat('reading data... ')
    if(!is.null(ddir2) && !is.null(datadir)) cat('(predict',setup2[1],'using',setup[1],'level0 models)')
    cat(' ', length(files), 'files\n')
    pb = txtProgressBar(min = 0, max = length(files), initial = 0, style = 3)
    for(i in 1:length(files)) {
        load(file.path(ddir, files[i]))

        itrain80 = result$itrain80
        itest20 = result$itest20

        ytrain = result$data$y[itrain80]
        ytest = result$data$y[itest20]
        Xtrain$mean[[i]] = result$data$mean[itrain80]
        Xtest$mean[[i]] = result$data$mean[itest20]
        Xtrain$age[[i]] = result$pred$validation[itrain80]
        stopifnot(!any(is.na(Xtrain$age[[i]])))
        Xtest$age[[i]] = apply(result$pred$test,1,mean)
        if(!is.null(ddir2) && !is.null(datadir)) {
            # we have a sample to predict
            parcel = file_path_sans_ext(basename(files[i]))
            # check if the cached prediction exists
            cachedir = file.path(cache,
                                paste(setupin, "-", setup2in, sep=""))
            dir.create(cachedir, showWarnings=FALSE, recursive=TRUE)
            cachefile = file.path(cachedir, paste(parcel, ".csv", sep=""))
            if(file.exists(cachefile)) {
                pp = read.csv(cachefile, header=TRUE)
            } else {
                # read in voxelwise data
                pfile = file.path(datadir, paste(parcel, '.csv', sep=''))
                xy = get_data_voxelwise(pfile, ycol, setup2[1])               
                # recreate the partition
                # get test indices for setup2
                seedfold2 = separate_seedfold(setup2[3], k=5)
                set.seed(seedfold2$seed)
                index = createFolds(xy$y, k=5, returnTrain = FALSE)[[seedfold2$fold]]
                if(finalModel) {
                    pp = predict(result$finalModel, xy$x[index,])
                } else {
                    pp = matrix(NA, nrow=length(index), ncol=length(result$models))
                    for(m in 1:length(result$models)) pp[,m] = predict(result$models[[m]], xy$x[index,])
                    pp = apply(pp,1,mean)
                }
                if(cacheagepred) write.csv(pp, file=cachefile, row.names=FALSE)
            }
            Xtest$agepred[[i]] = pp
        }
        setTxtProgressBar(pb, i)
    }
    close(pb)

    Xtrain$mean = do.call(cbind, Xtrain$mean)
    Xtrain$age = do.call(cbind, Xtrain$age)
    Xtest$mean = do.call(cbind, Xtest$mean)
    Xtest$age = do.call(cbind, Xtest$age)
    colnames(Xtrain$mean) = paste("X",1:length(files), sep="")
    colnames(Xtrain$age) = paste("X",1:length(files), sep="")
    colnames(Xtest$mean) = paste("X",1:length(files), sep="")
    colnames(Xtest$age) = paste("X",1:length(files), sep="")

    if(length(Xtest$agepred)>1) {
        Xtest$agepred = do.call(cbind, Xtest$agepred)
        colnames(Xtest$agepred) = paste("X",1:length(files), sep="")
    }

    return(list(Xtrain=Xtrain, Xtest=Xtest,
                ytrain=ytrain, ytest=ytest,
                itrain80=itrain80, itest20=itest20,
                ddir=ddir, ddir2=ddir2,
                finalModel=finalModel))
}

get_data_voxelwise <- function(file, ycol=NULL, data=NULL, datacol=1) {
    x = read.csv(file)
    parcel = file_path_sans_ext(basename(file))
    if(data!="all" && !is.null(data)) {
        datas = strsplit(data,'-')[[1]]
        ii = c()
        for(dd in 1:length(datas)) ii = c(ii, grep(datas[dd], x[,datacol]))
        x = x[ii,]
    }

    y = NULL
    if(!is.null(ycol)) y = x[,ycol]

    voxels = grep(paste(parcel,'_V',sep=''), colnames(x))

    colsuse = voxels
    if(!is.null(ycol)) colsuse = c(colsuse, grep(ycol, colnames(x)))

    return(list(x=data.matrix(x[,voxels]), y=y, z=x[,-colsuse], parcel=parcel))
}

level0CV <- function(x, y, folds=NULL, seed=NA, k=3, finalModel=TRUE,
                     preProc=c(), method='glmnet', result=list(),
                     xtest=NULL, ytest=NULL, itrain=NULL) {
    if(is.na(seed)) seed = sample(1:9999, 1)
    set.seed(seed)
    if(is.null(folds)) folds = createFolds(y, k=k, returnTrain=FALSE)
    else k = length(folds)
    result$folds = folds

    result$models = list() # one model per outer fold
    result$pred = list()
    result$pred$validation = rep(NA, length(y))
    if(!is.null(xtest) && !is.null(ytest)) {
        # one column per outer fold
        result$pred$test = matrix(NA, nrow=length(ytest), ncol=k)
    }
    for(ik in 1:k) {
        gc()
        #cat('fold', ik, 'of 3\n')
        ite = folds[[ik]]
        itr = setdiff(1:length(y), ite)
        result$models[[ik]] = train_model(y = y[itr], x = x[itr,],
                                    preProc = preProc,
                                    method = method)

        # predict the inner validation set, creates feature for meta-model
        result$pred$validation[ite] <- predict(result$models[[ik]], x[ite,])

        if(!is.null(xtest) && !is.null(ytest)) {
            # predict the outer test set, create features for the test set
            pred = predict_print(result$models[[ik]], xtest, ytest)
            result$pred$test[,ik] = pred$pred
        }
    }

    if(!is.null(itrain)) {
        n = length(y)
        if(!is.null(xtest) && !is.null(ytest)) n = n+length(ytest)
        predval = rep(NA, n)
        predval[itrain] = result$pred$validation
        result$pred$validation = predval
    }

    if(finalModel) {
        result$finalModel = train_model(y = y, x = x,
                                  preProc = preProc,
                                  method = method)
    }

    return(result)
}

five_point_dist <- function(x, y) {
    sqrt(sum((summary(x)-summary(y))^2))/5
}

get_oos_pred <- function(data1, data2, seed, method='glmnet', k=5,
                              resultdir=file.path('..', 'results', '80_20', 'pairs'),
                              ways=c('', 'agepred')) {
    predage = matrix(NA, ncol=3, nrow=0)
    colnames(predage) = c(data1, data2, 'y')
    predmean = predage
    for(ik in 1:k) {        
        setup1 = paste(data1, '_', method, '_', seed, '-', ik, sep='')
        setup2 = paste(data2, '_', method, '_', seed, '-', ik, sep='')

        resultfile = file.path(resultdir, paste(method,'_',setup1,'_',setup2,'.rda', sep=''))
        if(!file.exists(resultfile)) {
            tmp = setup1
            setup1 = setup2
            setup2 = tmp
            tmp = data1
            data1 = data2
            data2 = tmp
            resultfile = file.path(resultdir, paste(method,'_',setup1,'_',setup2,'.rda', sep=''))
        }
        setups = c(setup1, setup2)    
        
        cat('loading', resultfile, '\n')
        load(resultfile)
        for(way in ways) {
            # left setup is predictor and right setup is target            
            res = paste(setup1, setup1, sep='~')            
            p1 = result$predictions[[res]]$age$pred
            y1 = result$predictions[[res]]$age$y
            if(way=='agepred') res = paste(paste(setup2,'+agepred',sep=''), setup1, sep='~')
            else res = paste(setup2, setup1, sep='~')
            p2 = result$predictions[[res]]$age$pred
            y2 = result$predictions[[res]]$age$y
            stopifnot(length(y1)==length(y2))
            py = matrix(NA, nrow=length(y1), ncol=3)
            colnames(py) = c(data1, data2, 'y')
            py[,'y'] = y1
            py[, data1] = p1
            py[, data2] = p2
            predage = rbind(predage, py)

            if(way=='agepred') res = paste(paste(setup1,'+agepred',sep=''), setup2, sep='~')
            else res = paste(setup1, setup2, sep='~')
            p1 = result$predictions[[res]]$age$pred
            y1 = result$predictions[[res]]$age$y
            res = paste(setup2, setup2, sep='~')
            p2 = result$predictions[[res]]$age$pred
            y2 = result$predictions[[res]]$age$y
            stopifnot(length(y1)==length(y2))
            py = matrix(NA, nrow=length(y1), ncol=3)
            colnames(py) = c(data1, data2, 'y')
            py[,'y'] = y1
            py[, data1] = p1
            py[, data2] = p2
            predage = rbind(predage, py)
        }
    }
    return(list(age=predage, mean=predmean))
}

get_coef_models_level0 <- function(site, method='glmnet', atlas=NULL, saveit=TRUE, ...) {
    stopifnot(method %in% c('glmnet', 'glmnetnonpos', 'glmnetnonneg', 'ridge')) # only glmnet implemented for now
    library(glmnet) # needed for `coef` call

    ml = get_models_level0(site, method=method, ...)
    coefs = lapply(ml, function(x) as.matrix(coef(x$finalModel, 
            x$finalModel$lambdaOpt))[,1])
    names(coefs) = names(ml)

    if(saveit) {
        odir = file.path('..', 'results', 'coefs')
        dir.create(odir, recursive=TRUE, showWarnings=FALSE)
        ofile = file.path(odir, paste(method, '_', site, '_coefs.txt', sep=''))
        cat('saving coefs to ', ofile, '\n')        
        write.table(unlist(coefs), ofile, sep='\t', row.names=TRUE, col.names=FALSE)

        if(!is.null(atlas)) {
            # some models will not have all voxels due to nzv
            library(RNifti)
            library(oro.nifti)
            library(neurobase)
            cat('creatinhg nifti...\n')
            atl = readNIfTI(atlas)
            d = c(atl)
            nd = paste('x', 1:length(d), sep='')
            for(i in 1:length(coefs)) {
                j = which(d==i)
                nd[j] = paste(paste('P',i,sep=''),
                        paste('V', 1:length(j), sep=''), sep='_')
                cf = coefs[[i]][-c(1)]
            }
            d = rep(NA, length(d))
            names(d) = nd

            cf = unlist(lapply(coefs, function(x) x[-c(1)]))
            names(cf) = sapply(names(cf), function(x) strsplit(x, '\\.')[[1]][2])
            stopifnot(all(names(cf) %in% nd))
            d[names(cf)] = cf
            #atl = asNifti(d, reference=atl)
            ofile = file.path(odir, paste(method, '_', site, '_coefs', sep=''))
            cat('saving nifti to ', ofile, '\n')
            #not sure where the a below is coming from
            a = write_nifti(a, atl, file)
        }
    }

    coefs
}

get_models_level0 <- function(site, method='glmnet', cl=NULL, 
                     cacheit=TRUE, trimit=TRUE,
                     cache=file.path('..', 'models', 'cache_level0')) {
    models = NULL
    cachefile = file.path(cache, site, paste(method, '.RData', sep=''))
    if(file.exists(cachefile)) {
        cat('loading cached models', cachefile, '\n')
        load(cachefile)
        return(models)
    }

    cachedir = file.path(cache, site, method)
    if(!dir.exists(cachedir)) return(models)
    files = list.files(cachedir, pattern='*.RData')
    if(length(files)==0) return(models)
    
    files = mixedsort(files)
    parcels = sapply(files, 
                     function(ff) file_path_sans_ext(basename(ff)))        
    cat('reading level0 models...',site,method,'\n')
    # read in all the level0 models
    # but also trim them to remove extra info from caret
    models = pblapply(files, function(ff, trimit) {
                cfile = file.path(cachedir, ff)
                load(cfile)
                if(trimit) return(trim.train(result$finalModel))
                else return(result$finalModel)
            }, trimit=trimit, cl=cl)
    names(models) = parcels

    if(cacheit) {
        cat('caching level0 models...',cachefile,'\n')
        save(models, file=cachefile)
    }

    return(models)
}

get_models_level1 <- function(site, method='glmnet', method0='glmnet',
                     cache=file.path('..', 'models', 'cache_level1')) {
    #' returns level1 models created from method and method0,
    #' if exist in cache
    result = NULL
    file = file.path(cache, paste(method, '_', method0, '_', site, '.rda', sep=''))
    if(file.exists(file)) {
        cat('loading cached models', file, '\n')
        load(file)
    }
    return(result)
}

# https://github.com/topepo/caret/blob/master/pkg/caret/R/trim.R
trim.train <- function(object, ...) {
  removals <- c("results", "pred", "bestTune", "call", "dots",
                "metric", "trainingData", "resample", "resampledCM",
                "perfNames", "maxmimize", "times")
  for(i in removals)
    if(i %in% names(object)) object[i] <- NULL
  c_removals <- c('method', 'number', 'repeats', 'p', 'initialWindow', 
                  'horizon', 'fixedWindow', 'verboseIter', 'returnData', 
                  'returnResamp', 'savePredictions', 'summaryFunction', 
                  'selectionFunction', 'index', 'indexOut', 'indexFinal',
                  'timingSamps', 'trim', 'yLimits')
  for(i in c_removals)
    if(i %in% names(object$control)) object$control[i] <- NULL  
  if(!is.null(object$modelInfo$trim))
    object$finalModel <- object$modelInfo$trim(object$finalModel)
  object
}


get_oos_level1 <- function(sites, l0_oos_from_test=TRUE,
                           cache=file.path('..','models','cache_level1')) {
    #l0_oos_from_test
    # FLASE: use OOS age predictions from train->test site, test site does not have enough data for l0 models
    # TRUE: use OOS predictions from the test site, the test site has enough data for l0 models
    # generate out-of-sample data for level2 models
    oos = list()
    oos$age = matrix(NA, ncol=length(sites), nrow=0)
    colnames(oos$age) = sites
    oos$mean = oos$age
    oos$y = c()
    for(sitetest in sites) {
        # sitetest is the test site
        cat('loading data for test site', sitetest, '\n')
        resulttest = get_models_level1(sitetest, cache=cache)
        stopifnot(!is.null(resulttest))
        datamean = resulttest$data$mean
        if(is.null(colnames(datamean))) colnames(datamean) = 1:ncol(datamean)
        dataage  = resulttest$data$pred
        if(is.null(colnames(dataage))) colnames(dataage) = 1:ncol(dataage)
        sitey    = resulttest$data$y
        # do CV
        cat('doing CV for site', sitetest, '...\n')
        cat('CV on datamean\n')
        cvmean = level0CV(datamean, sitey)
        cat('CV on dataage\n')
        cvage  = level0CV(dataage, sitey)
        # get oos predictions on all sites
        oosm = matrix(NA, ncol=length(sites), nrow=length(sitey))
        oosa = matrix(NA, ncol=length(sites), nrow=length(sitey))
        colnames(oosm) = sites
        colnames(oosa) = sites
        for(sitetrain in sites) {
            # sitetrain is the training site
            # use CV when same site
            if(sitetest==sitetrain) {
                oosm[,sitetrain] = cvmean$pred$validation
                oosa[,sitetrain] = cvage$pred$validation
            } else {
                cat('loading data for train site', sitetrain, '\n')
                resulttrain = get_models_level1(sitetrain, cache=cache)
                stopifnot(!is.null(resulttrain))
                modelmean = resulttrain$models$mean
                modelage  = resulttrain$models$age
                oosm[,sitetrain] = predict(modelmean, datamean)
                if(l0_oos_from_test) {
                    cat('oos data from within test', sitetest, '\n')
                } else {
                    cat('oos data with train', sitetrain, 'and test', sitetest, '\n')
                    oosl0 = get_oos_level0(sitetrain, sitetest, cache1=cache)
                    stopifnot(all(oosl0$y==sitey))
                    dataage = oosl0$age
                }
                oosa[,sitetrain] = predict(modelage, dataage)
            }
        }
        oos$y = c(oos$y, sitey)
        oos$mean = rbind(oos$mean, oosm)
        oos$age  = rbind(oos$age, oosa)
    }
    colnames(oos$mean) = sites
    colnames(oos$age) = sites
    return(oos)
}

# predicts site test data using level0 pretrained models
get_oos_level0 <- function(sitetrain, sitetest, datadir=file.path('..','data'),
                           cache0=file.path('..','models','cache_level0'),
                           cache1=file.path('..','models','cache_level1'),
                           method='glmnet', ycol='age', cacheit=TRUE) {

    ofile=file.path(cache0, paste('l0_', method, '_train_', sitetrain, '_test_', sitetest, '.RData', sep=''))
    if(file.exists(ofile)) {
        cat('loading cached oos data', ofile, '\n')
        load(ofile)
        return(data)
    }

    # generate out-of-sample data for level1 models
    data = list()
    files = list.files(datadir, pattern='*.csv')
    files = mixedsort(files)
    data$age  = vector(mode = "list", length = length(files))

    cat('predicting',sitetest,'using',sitetrain,'level0 models...\n')
    cat('getting level0 models\n')
    # call trained models of level0
    ml0 = get_models_level0(sitetrain, method=method, cache=cache0)

    pb = txtProgressBar(min = 0, max = length(files), style = 3)
    # each file is a parcel (873 here)
    for(i in 1:length(files)) {
        parcel = file_path_sans_ext(basename(files[i]))
        pfile = file.path(datadir, files[i])
        x = get_data_voxelwise(pfile, ycol=ycol, data=sitetest)
        data$age[[i]] = predict(ml0[[parcel]], x$x)
        setTxtProgressBar(pb, i)
    }
    close(pb)
    data$y = x$y
    ml0 = NULL # save memory
    data$age  = do.call(cbind, data$age)
    colnames(data$age)  = 1:ncol(data$age)

    if(cacheit) {
        cat('caching level0 predictions...', ofile, '\n')
        save(data, file=ofile)
    }
}

get_pred_level0 <- function(site, method='glmnet',
                            cache=file.path('..','models','cache_level0'), cl=NULL) {
    cachedir = file.path(cache, site, method)
    cat('get_pred_level0 cachedir', cachedir, '\n')
    stopifnot(dir.exists(cachedir))
    files = list.files(cachedir, pattern='*.RData')
    stopifnot(length(files)>0)
    files = mixedsort(files)
    res = pblapply(files, function(f, cachedir) {
        result = NULL
        load(file.path(cachedir, f))
        return(result$pred$validation)
    }, cachedir=cachedir, cl=cl)
    res = do.call(cbind, res)
    stopifnot(ncol(res)== length(files))
    colnames(res) = 1:length(files)
    return(res)
}

get_parcelwise_corr <- function(data, method='glmnet', result=NULL,
                                cache=file.path('..','models','cache_level0'),
                                atlas=NULL, subset=NULL, suffix='') {
    # get parcelwise correlation
    if(!is.null(result)) {
        cc = result$corr
        pv = result$p.value
    } else {
        files = list.files(file.path(cache, data, method), pattern='*.RData')
        files = mixedsort(files)
        cc = rep(NA, length(files))
        pv = rep(NA, length(files))
        cat('getting parcelwise correlation...\n')
        pb = txtProgressBar(min = 0, max = length(files), style = 3)
        for(i in 1:length(files)) {
            parcel = file_path_sans_ext(basename(files[i]))
            load(file.path(cache, data, method, files[i]))
            if(is.null(subset)) subset = rep(TRUE, length(result$data$y))
            ct = cor.test(result$data$y[subset], result$pred$validation[subset])
            cc[i] = ct$estimate
            pv[i] = ct$p.value
            setTxtProgressBar(pb, i)
        }
        close(pb)
    }

    if(!is.null(atlas)) {
        cat('creating niftis...\n')
        odir = file.path('..', 'results', 'coefs')
        library(RNifti)
        library(oro.nifti)
        library(neurobase)
        atl = readNIfTI(atlas)
        parcels = c(atl)

        ofile = file.path(odir, paste(method, '_', data, '_parcelwise_corr', suffix, sep=''))
        a = assign_values_to_parcels(parcels, cc)
        a = write_nifti(a, atl, ofile)

        ofile = file.path(odir, paste(method, '_', data, '_parcelwise_pval', suffix, sep=''))
        a = assign_values_to_parcels(parcels, pv)
        a = write_nifti(a, atl, ofile)

        ofile = file.path(odir, paste(method, '_', data, '_parcelwise_corr_fdr', suffix, sep=''))
        ca = cc
        ca[p.adjust(pv, method = 'fdr') > 0.05] = 0
        a = assign_values_to_parcels(parcels, ca)
        a = write_nifti(a, atl, ofile)
    }

    return(list(data=data, corr=cc, p.value=pv))
}

assign_values_to_parcels <- function(parcels, values) {
    uparcels = unique(parcels)
    uparcels = uparcels[!is.na(uparcels) & uparcels>0]
    uparcels = sort(uparcels)
    stopifnot(length(uparcels)==length(values))
    assigned = rep(NA, length(parcels))
    for(i in 1:length(uparcels)) {
        p = uparcels[i]
        assigned[parcels==p] = values[i]
    }
    assigned
}

write_nifti <- function(x, ref, file, datatype=16) {
    library(RNifti)
    library(oro.nifti)
    library(neurobase)
    a = nifti(array(x, dim=dim(ref)), datatype=datatype)
    a = copyNIfTIHeader(ref, a)
    cat('saving nifti to ', file, '\n')
    writeNIfTI(a, filename=file, verbose=TRUE)
    return(a)
}

harmonize_covbat <- function(xtrain, xtest=NULL, modtrain=NULL, modtest=NULL) {
    # xtrain and xtest and modtrain should be lists
    # there should be one site per element
    stopifnot(is.list(xtrain))
    if(!is.null(xtest)) stopifnot(is.list(xtest))

    if(is.null(names(xtrain))) names(xtrain) = 1:length(xtrain)
    if(!is.null(xtest) && is.null(names(xtest))) names(xtest) = 1:length(xtest)
    if(!is.null(modtrain)) {
        stopifnot(length(modtrain)==length(xtrain))
        if(is.null(names(modtrain))) names(modtrain) = 1:length(modtrain)
    }
    if(!is.null(modtest)) {
        stopifnot(length(modtest)==length(xtest))
        if(is.null(names(modtest))) names(modtest) = 1:length(modtest)
    }

    cat('\nharmonizing using CovBat\n')
    cat('\ttrain batches', length(xtrain),'\n')
    if(!is.null(xtest)) cat('\ttest batches', length(xtest),'\n')

    datidx = c()
    bat = c()
    XX = c()
    namestrain = names(xtrain)
    for(i in 1:length(xtrain)) {
        bat = c(bat, rep(namestrain[i], nrow(xtrain[[i]])))
        XX = rbind(XX, xtrain[[i]])
    }
    datidx = paste("train", bat, sep="")

    if(!is.null(xtest)) {
        namestest = names(xtest)
        battest = c()
        for(i in 1:length(xtest)) {
            battest = c(battest, rep(namestest[i], nrow(xtest[[i]])))
            XX = rbind(XX, xtest[[i]])
        }
        bat = c(bat, battest)
        datidx = c(datidx, paste("test", battest, sep=""))
    }
    trainidx = rep(FALSE, length(bat))
    trainidx[grep("train", datidx)] = TRUE
    show(sort(table(datidx), decreasing = TRUE))

    # deal with the covariates
    mod = NULL
    if(!is.null(modtrain)) {
        cat('using mod in harmonization\n')
        mod = do.call(rbind, modtrain)
        if(!is.null(modtest)) {
            mod = rbind(mod, do.call(rbind, modtest))
        } else {
            ntest = sum(trainidx==FALSE)
            mod = rbind(mod, matrix(NA, nrow=ntest, ncol=ncol(mod)))
        }
        if(is.null(colnames(mod))) colnames(mod) = paste('x', 1:ncol(mod), sep='')
        mod = stats::model.matrix(as.formula(paste('~', paste(colnames(mod), collapse='+'))), 
                                  data=as.data.frame(mod))
    }

    # covbat needs features in rows
    # only use the training samples to train
    XX  = covbat(dat=t(XX), bat=bat, train=trainidx, mod=mod)
    XX$dat.covbat = t(XX$dat.covbat)

    xtrain = list()
    for(i in 1:length(namestrain)) {
        xtrain[[namestrain[i]]] = XX$dat.covbat[grep(paste("train",namestrain[i],sep=""), datidx),]
    }
    if(!is.null(xtest)) {
        xtest = list()
        for(i in 1:length(namestest)) {
            xtest[[namestest[i]]] = XX$dat.covbat[grep(paste("test",namestest[i],sep=""), datidx),]
        }
    }

    return(list(covbat=XX, bat=bat, datidx=datidx, trainidx=trainidx, xtrain=xtrain, xtest=xtest))
}


get_data_parcelwise <- function(site, method='glmnet',
                  cache=file.path('..', 'models', 'cache_level0')) {
    cache = file.path(cache, site, method)
    files = list.files(cache, pattern='*.RData')
    files = mixedsort(files)
    data = list()
    data$files = files
    data$mean = list()
    data$median = list()
    data$pred = list()
    pb = txtProgressBar(min = 0, max = length(files), style = 3)
    for(i in 1:length(files)) {
        pfile = file.path(cache, files[i])
        load(pfile)
        data$y = result$data$y
        data$mean[[i]] = result$data$mean
        data$median[[i]] = result$data$median
        data$pred[[i]] = result$pred$validation
        setTxtProgressBar(pb, i)
    }
    close(pb)
    data$mean = do.call(cbind, data$mean)
    colnames(data$mean) = 1:length(files)
    data$median = do.call(cbind, data$median)
    colnames(data$median) = 1:length(files)
    data$pred = do.call(cbind, data$pred)
    colnames(data$pred) = 1:length(files)
    return(data)
}
