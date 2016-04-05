set autoscale
unset log
unset label
set xtic 1000
set ytic 0.05
set title "Training"
set xlabel "Iteration"
set ylabel "Accuracy"
set key right bottom
set xrange []
set yrange [0:]
set grid
set terminal wxt size 1300,600
set key autotitle columnhead
plot "out.csv" using 1:2 with linespoints pt 7, \
"out.csv" using 1:6  with linespoints pt 7, \
"out.csv" using 1:10 with linespoints pt 7, \
"out.csv" using 1:14  with linespoints pt 7, \
"out.csv" using 1:18  with linespoints pt 7, \
"out.csv" using 1:22  with linespoints pt 7
pause -1
plot "out.csv" using 1:4 with linespoints pt 7, \
"out.csv" using 1:8  with linespoints pt 7, \
"out.csv" using 1:12 with linespoints pt 7, \
"out.csv" using 1:16  with linespoints pt 7, \
"out.csv" using 1:20  with linespoints pt 7, \
"out.csv" using 1:24  with linespoints pt 7
pause -1
