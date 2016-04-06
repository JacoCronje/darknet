set autoscale
unset log
unset label
set xtic 10000
set ytic 0.05
set title "Training MSE"
set xlabel "Iteration"
set ylabel "mse"
set key right bottom
set xrange []
set yrange [:1]
set grid
set terminal wxt size 1300,600
set key autotitle columnhead
plot "out.csv" using 1:4 with linespoints pt 7, \
"out.csv" using 1:10  with linespoints pt 7, \
"out.csv" using 1:16 with linespoints pt 7, \
"out.csv" using 1:22  with linespoints pt 7, \
"out.csv" using 1:28  with linespoints pt 7, \
"out.csv" using 1:34  with linespoints pt 7
pause -1
