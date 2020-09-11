lms1 = 0.10
rms1 = 0.74
lms2 = 0.75
rms2 = 0.96
bms  = 0.11
offsetylabel = -2.0
sizex = 7.5
sizey = 5
normalize = 4e-6

set tics font "Consolas,18"
set xlabel font "Consolas,18"
set ylabel font "Consolas,18"

set terminal pdf enhanced color size sizex in, sizey in
set output "scatter_bar.pdf"

binwidth=0.01
bin(x,width)=width*floor(x/width)+width/2.0
set table "histogram.txt"
plot "data.csv" u (bin(($2/normalize),binwidth)):(1.0) smooth freq with boxes
unset table

set ytics binwidth*4
set format y "%.1f"

set multiplot

set lmargin screen lms1
set rmargin screen rms1
set bmargin screen bms

set xtics 0, 100, 600
set xlabel "{/Consolas:Italic h} []"
set ylabel "{/Consolas:Italic f} []" offset offsetylabel,0
set xrange[0:650]
set yrange[0:3.5]
plot "data.dat" u ($1/normalize):($2/normalize) with points ti "" pt 1 ps 0.5 lc "blue"


set lmargin screen lms2
set rmargin screen rms2

set xrange[0:*]
set yrange[0:3.5]
set xtics 70
set format y ""
set ylabel ""
set xlabel "Count []"
set style fill solid border lc rgb "black"
plot "histogram.txt" u ($2*0.5):($1):($2*0.5):(binwidth*0.5) with boxxyerrorbars lw 0.5 lc rgb "light-blue" ti ""

unset multiplot

set output
