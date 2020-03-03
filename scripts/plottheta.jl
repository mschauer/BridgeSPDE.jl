using Plots
using DelimitedFiles
using FileIO
pgfplotsx()
thetas = readdlm("thetas.txt")
p = plot(0:45, thetas, color=:black, ylims = [(0,2.5) (0, 9)], layout = (2,1), label=["\$\\theta_1\$" "\$\\theta_2\$"]);
vspan!(p, [[0,15] [0,15]],
        color = :black, alpha = 0.15,
        labels = ["20 x 24" "20 x 24"])
        #; "41 x 49" "55 x 65"
vspan!(p, [[15,30] [15,30]], color = :white, alpha = 0.15, labels = ["41 x 49" "41 x 49"]);
vspan!(p, [[30,45] [30,45]], color =  :blue, alpha = 0.15, labels = "55 x 65");

p

save("thetas.png", p)

@show thetas
