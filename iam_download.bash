#!/bin/bash
function make_folder () {
    mkdir NonSimpleHTR/data
    mkdir NonSimpleHTR/data/words
}

function download_data () {
    local current_location=pwd
    cd NonSimpleHTR/data/words
    wget http://www.fki.inf.unibe.ch/DBs/iamDB/data/words/words.tgz --user jalotra --password shivam567
    tar -xvf words.tgz
    cd .. 
    wget http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/words.txt --user jalotra --password shivam567
    # Return to original location 
    cd $current_location
}

make_folder
download_data
