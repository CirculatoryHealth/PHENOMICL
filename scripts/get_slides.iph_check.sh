#!/usr/bin/env bash

# author: Sander W. van der Laan | s.w.vanderlaan-2@umcutrecht.nl
# last update: 2023-11-28

### Creating display functions
### Setting colouring
NONE='\033[00m'
BOLD='\033[1m'
# OPAQUE='\033[2m'
FLASHING='\033[5m'
# UNDERLINE='\033[4m'

RED='\033[01;31m'
# GREEN='\033[01;32m'
# YELLOW='\033[01;33m'
# PURPLE='\033[01;35m'
CYAN='\033[01;36m'
# WHITE='\033[01;37m'
### Regarding changing the 'type' of the things printed with 'echo'
### Refer to: 
### - http://askubuntu.com/questions/528928/how-to-do-underline-bold-italic-strikethrough-color-background-and-size-i
### - http://misc.flogisoft.com/bash/tip_colors_and_formatting
### - http://unix.stackexchange.com/questions/37260/change-font-in-echo-command

### echo -e "\033[1mbold\033[0m"
### echo -e "\033[3mitalic\033[0m" ### THIS DOESN'T WORK ON MAC!
### echo -e "\033[4munderline\033[0m"
### echo -e "\033[9mstrikethrough\033[0m"
### echo -e "\033[31mHello World\033[0m"
### echo -e "\x1B[31mHello World\033[0m"

# for i in $(seq 0 5) 7 8 $(seq 30 37) $(seq 41 47) $(seq 90 97) $(seq 100 107) ; do 
# 	echo -e "\033["$i"mYou can change the font...\033[0m"; 
# done
### Creating some function
# function echobold { #'echobold' is the function name
#     echo -e "\033[1m${1}\033[0m" # this is whatever the function needs to execute.
# }
function echocyan { #'echobold' is the function name
    echo -e "${CYAN}${1}${NONE}" # this is whatever the function needs to execute.
}
function echobold { #'echobold' is the function name
    echo -e "${BOLD}${1}${NONE}" # this is whatever the function needs to execute, note ${1} is the text for echo
}
function echoitalic { #'echobold' is the function name
    echo -e "\033[3m${1}\033[0m" # this is whatever the function needs to execute.
}
function echowarning { #'echobold' is the function name
    echo -e "${FLASHING}${RED}${1}${NONE}" # this is whatever the function needs to execute.
}

# Help function
function show_help {
  echocyan "Usage: $0 [-s|--stain STAIN]"
  echocyan "Copy whole-slide images (WSI) for a specific stain from bulk to virtual slides directory."
  echocyan
  echocyan "Options:"
  echocyan "  -s, --stain STAIN    Specify the stain type (e.g., CD14, CD3, EVG, etc.). Default is HE."
  echocyan "  -h, --help           Display this help and exit."
  echocyan
  echocyan "Description:"
  echocyan "This script copies whole-slide images (WSI) for a specified stain from a bulk directory"
  echocyan "to a virtual slides directory. It uses a list of sample numbers for CEA patients and"
  echocyan "performs the copy operation for each sample. The stain type can be customized using the -s or --stain option."
  exit 0
}

# Here we set the list of samples selected for the analyses
# These are 100 patients for manual checking of AI-based IPH scoring
SAMPLE_LIST="158 1963 2726 144 2404 105 2061 1531 2417 492 234 1258 3777 3156 2670 1296 2385 860 1550 2685 186 1161 247 491 308 3741 3773 646 480 1329 1657 3888 661 1795 231 3291 481 2228 748 754 919 321 2465 637 1412 2055 251 1728 3175 3690 730 1735 1265 2862 2173 1850 1747 528 2810 250 1352 1654 3433 2016 3270 365 1672 791 1222 3757 2948 2976 878 487 87 756 844 2227 1751 2073 761 248 565 297 3724 135 561 3162 35 1702 660 1166 76 3216 24 688 932 1286 312 173"

# Here we set some directories
BULKDIR="/data/isi/d/dhl/ec/VirtualSlides/AE-SLIDES/"
VIRTUALSLIDESDIR="/hpc/dhl_ec/VirtualSlides"
WORKDIR="IPH_CHECK"

# Here we copy for a given stain the slides from the bulk directory to the virtual slides directory
# Stains are:
    # CD34
    # CD66b
    # CD68
    # EVG
    # FIBRIN
    # GLYCC
    # HE
    # SMA
    # SR
    # SR_POLARIZED

# Check if any command line options are provided
if [ "$#" -eq 0 ]; then
  echo "Error: No options provided. Use -h or --help for usage information."
  exit 1
fi

# Command line options
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -s|--stain)
      STAIN="$2"
      shift
      ;;
    -h|--help)
      show_help
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

# Set default stain if not provided
STAIN="${STAIN:-HE}"

echo "Creating the [$WORKDIR] directory in the [$VIRTUALSLIDESDIR] folder."
if [ ! -d $VIRTUALSLIDESDIR/$WORKDIR ]; then
	mkdir $VIRTUALSLIDESDIR/$WORKDIR
else
	echo "[$WORKDIR] directory exists -- skipping"
fi
echo "Creating the [$STAIN] directory in the [$WORKDIR] folder."
if [ ! -d $VIRTUALSLIDESDIR/$WORKDIR/$STAIN ]; then
	mkdir $VIRTUALSLIDESDIR/$WORKDIR/$STAIN
else
	echo "[$WORKDIR/$STAIN] directory exists -- skipping"
fi

echo "Start the copying of WSI for [$STAIN]."
for STUDYNUMBER in $SAMPLE_LIST; do

    echo "> Copying slides for studynumber [AE${STUDYNUMBER}] and stain [${STAIN}]"
    ### Uncomment the following line to perform the actual copy
    cp -v $BULKDIR/$STAIN/AE${STUDYNUMBER}.* $VIRTUALSLIDESDIR/$WORKDIR/$STAIN/
    ### reference to https://www.digitalocean.com/community/tutorials/how-to-use-rsync-to-sync-local-and-remote-directories
    ### --delete: In order to keep two directories truly in sync, it’s necessary to delete files 
    ###         from the destination directory if they are removed from the source. 
    ###         By default, rsync does not delete anything from the destination directory.
    ### --partial: Tells rsync to keep partially transferred files (and upon resume 
    ###         rsync will use partially transferred files always after checksumming safely)
    ### --progress: This option tells rsync to print information showing the progress of the transfer.
    ### --human-readable/-h: Output numbers in a more human-readable format.
    ### --archive/-a: This is equivalent to -rlptgoD. It is a quick way of saying you want recursion 
    ###         and want to preserve almost everything (with -H being a notable omission). 
    ###         The only exception to the above equivalence is when --files-from is specified, 
    ###         in which case -r is not implied.
    ### --verbose/-v: This option increases the amount of information you are given during the transfer.
    ### --compress/-z: With this option, rsync compresses the file data as it is sent to the destination machine, 
    ###         which reduces the amount of data being transmitted -- something that is useful over a slow connection.
    # rsync -avz --progress --partial --delete -h $BULKDIR/$STAIN/AE${STUDYNUMBER}.* $VIRTUALSLIDESDIR/$WORKDIR/$STAIN/

done


