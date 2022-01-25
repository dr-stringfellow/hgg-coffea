touch faults.txt
rm -f faults.txt
touch lines.txt
rm -f lines.txt
touch workers.html
rm -f workers.html

port=`lsof -i -P | grep T3DESK012.MIT.EDU | grep LISTEN | awk '{print $9}' | awk -F':' '{print $2}'`
#echo $port

python resub.py $port > webpage
source webpage
wget $webpage
grep -nr 'progress' workers.html | sed 's/:.*//g' > lines.txt

while read p;
do
    newline=$((p+2))
    #echo $newline                                                                                                             
    faultylines=`awk "NR == ${newline}" workers.html | awk '{print $2}'`
    #echo $newline                                                                                                             
    #echo $faultylines                                                                                                         
    tmp=`echo $newline $faultylines | grep -v ' 0'`
    if [ ! -z "$tmp" ]
    then
        echo "hi"
        #echo $tmp                                                                                                             
        host=$((p-4))
        ips=`awk "NR == ${host}" workers.html | sed 's/.*\/\///g' | sed 's/<.*//g'`
        echo tcp://$ips
        echo tcp://$ips >> faults.txt
    fi
done < lines.txt

python resub.py $port faults.txt

