#/bash/sh 
set -eu

rm -f data_pressure.hist
hist="data_pressure.hist"
tmp=tmp.sh

grid="../recflows/grid_z0003"
flow="../recflows"
nst=95100
nit=100
nls=114800
iz=3

echo "ReadGrid "$grid >>$hist
echo "CreateWindow -fix	" >>$hist 
#echo "FullView 1 0 0 0 -0 -1 -0 0 0 1 33.6806 -2 0 0 0.255" >>$hist #cylinder
echo "FullView 1 0 0 0 -0 -1 -0 0 0 1 9.43023 -2 0 0 0.25" >>$hist
echo "ViewDirection 1 -Y" >>$hist
echo "RangeMode ByHand" >>$hist
echo "rf "$flow/recflow_z0${iz}_00095100 >>$hist
echo "fn pressure" >>$hist
echo "cfs k=1" >>$hist
echo "cr 0.94 1.02" >>$hist
echo %mkdir $flow/pngs >>$hist
echo %import -window \"postkun 1\" ./$flow/pngs/`printf 00095100`.png >>$hist

for n in $(seq $nst $nit $nls);
do
	echo "rf "$flow/recflow_z0${iz}_`printf "%08d\n" $n`  >>$hist
	echo "fn pressure" >>$hist
	echo %import -window \"postkun 1\" ./$flow/pngs/`printf "%08d\n" $n`.png >>$hist
done

# open post kun
echo \#!/bin/sh >$tmp
echo vglrun -ms 8 /usr/local/bin/post \<\< GNU >>$tmp                    
echo \#$hist -p >>$tmp                                  
echo exit >>$tmp                 
echo GNU >>$tmp                                                          
sh $tmp 
