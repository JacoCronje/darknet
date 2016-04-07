set autoscale
unset log
unset label
set xtic 5000
set ytic 0.01
set title "Training MSE"
set xlabel "Iteration"
set ylabel "mse"
set key right top
set xrange [35000:]
set yrange [:1]
set grid
set terminal wxt size 1200,800
set key autotitle columnhead
plot "out.csv" using 1:4 with linespoints pt 7, \
"out.csv" using 1:10  with linespoints pt 7, \
"out.csv" using 1:16 with linespoints pt 7, \
"out.csv" using 1:22  with linespoints pt 7, \
"out.csv" using 1:28  with linespoints pt 7, \
"out.csv" using 1:34  with linespoints pt 7, \
"out.csv" using 1:40  with linespoints pt 7, \
"out.csv" using 1:46  with linespoints pt 7
pause -1
