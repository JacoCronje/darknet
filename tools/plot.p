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
set yrange [0:1]
set grid
set terminal wxt size 1300,600
plot "out.csv" using 1:2 title 'shrink_add' with linespoints pt 7, \
"out.csv" using 1:6 title 'compact' with linespoints pt 7, \
"out.csv" using 1:10 title 'shrink_add_64' with linespoints pt 7, \
"out.csv" using 1:12 title 'tst_shrink_add_64' with linespoints pt 7
#"out.csv" using 1:4 title 'tst_shrink_add' with linespoints pt 7, \
#"out.csv" using 1:8 title 'tst_compact' with linespoints pt 7
pause -1
