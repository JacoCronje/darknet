set autoscale
unset log
unset label
set xtic 5000
set ytic 0.01
set title "Testing"
set xlabel "Iteration"
set ylabel "Accuracy"
set key right bottom
set xrange [35000:]
set yrange [:]
set grid
set terminal wxt size 1200,800
set key autotitle columnhead
plot "out.csv" using 1:5 with linespoints pt 7, \
"out.csv" using 1:11  with linespoints pt 7, \
"out.csv" using 1:17 with linespoints pt 7, \
"out.csv" using 1:23  with linespoints pt 7, \
"out.csv" using 1:29  with linespoints pt 7, \
"out.csv" using 1:35  with linespoints pt 7, \
"out.csv" using 1:41  with linespoints pt 7, \
"out.csv" using 1:47  with linespoints pt 7
pause -1
