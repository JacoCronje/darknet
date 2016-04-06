set autoscale
unset log
unset label
set xtic 10000
set ytic 0.05
set title "Testing MSE"
set xlabel "Iteration"
set ylabel "mse"
set key right bottom
set xrange []
set yrange [:1]
set grid
set terminal wxt size 1300,600
set key autotitle columnhead
plot "out.csv" using 1:7 with linespoints pt 7, \
"out.csv" using 1:13  with linespoints pt 7, \
"out.csv" using 1:19 with linespoints pt 7, \
"out.csv" using 1:25  with linespoints pt 7, \
"out.csv" using 1:31  with linespoints pt 7, \
"out.csv" using 1:37  with linespoints pt 7
pause -1
