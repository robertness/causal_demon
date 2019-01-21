file <- 'http://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz'
isachs <- readr::read_delim(file, delim = " ")
devtools::use_data(isachs)
