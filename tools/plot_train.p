set autoscale
unset log
unset label
set xtic 5000
set ytic 0.01
set title "Training"
set xlabel "Iteration"
set ylabel "Accuracy"
set key right bottom
set xrange [35000:]
set yrange [:]
set grid
set terminal wxt size 1200,800
set key autotitle columnhead
plot "out.csv" using 1:2 with linespoints pt 7, \
"out.csv" using 1:8  with linespoints pt 7, \
"out.csv" using 1:14 with linespoints pt 7, \
"out.csv" using 1:20  with linespoints pt 7, \
"out.csv" using 1:26  with linespoints pt 7, \
"out.csv" using 1:32  with linespoints pt 7, \
"out.csv" using 1:38  with linespoints pt 7, \
"out.csv" using 1:44  with linespoints pt 7
pause -1

