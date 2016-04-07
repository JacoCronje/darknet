set autoscale
unset log
unset label
set xtic 5000
set ytic 0.01
set title "Testing MSE"
set xlabel "Iteration"
set ylabel "mse"
set key right top
set xrange [35000:]
set yrange [:1]
set grid
set terminal wxt size 1200,800
set key autotitle columnhead
plot "out.csv" using 1:7 with linespoints pt 7, \
"out.csv" using 1:13  with linespoints pt 7, \
"out.csv" using 1:19 with linespoints pt 7, \
"out.csv" using 1:25  with linespoints pt 7, \
"out.csv" using 1:31  with linespoints pt 7, \
"out.csv" using 1:37  with linespoints pt 7, \
"out.csv" using 1:43  with linespoints pt 7, \
"out.csv" using 1:49  with linespoints pt 7
pause -1
