## Calls the brainwise_multisite function for various combinations of the given datasets
##
## Author: Kaustubh R. Patil
## Date: Jan 2023

datas = c('1000Gehirne', 'CamCAN', 'eNKI', 'IXI')

for(d1 in datas) {
    for(d2 in datas) {
        # combine datasets. First in pairs and then for datas -(d1, d2)  
        if(d1 != d2) {
            xx = brainwise_multisite(sites=c(d2), newdata=d1, newdatadir='../data')
            dx = setdiff(datas, c(d1, d2))
            xx = brainwise_multisite(sites=c(dx), newdata=d1, newdatadir='../data')
        }
    }
    # finally for datas -(d1)
    dx = setdiff(datas, d1)
    xx = brainwise_multisite(sites=c(dx), newdata=d1, newdatadir='../data')
}


