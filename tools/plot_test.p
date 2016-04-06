set autoscale
unset log
unset label
set xtic 10000
set ytic 0.05
set title "Testing"
set xlabel "Iteration"
set ylabel "Accuracy"
set key right bottom
set xrange []
set yrange [0:]
set grid
set terminal wxt size 1300,600
set key autotitle columnhead
plot "out.csv" using 1:5 with linespoints pt 7, \
"out.csv" using 1:11  with linespoints pt 7, \
"out.csv" using 1:17 with linespoints pt 7, \
"out.csv" using 1:23  with linespoints pt 7, \
"out.csv" using 1:27  with linespoints pt 7, \
"out.csv" using 1:33  with linespoints pt 7
pause -1
