#!/bin/bash
function make_folder () {
    mkdir CleanNonSimpleHTR/data
    mkdir CleanNonSimpleHTR/data/words
}

function download_data () {
    local current_location=pwd
    cd CleanNonSimpleHTR/data/words
    wget http://www.fki.inf.unibe.ch/DBs/iamDB/data/words/words.tgz --user your_username --password your_password
    tar -xvf words.tgz
    cd .. 
    wget http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/words.txt --user your_username --password your_password
    # Return to original location 
    cd $current_location
}

make_folder
download_data
