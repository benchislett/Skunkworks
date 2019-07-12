# Second session, 10am-12pm, Friday July 12, 2019

using Statistics
using DelimitedFiles
using DataFrames
using CSV
using Printf
using Plots

x = [1,2,3,4] # Row Vectors
y = collect(1:2:10) # Ranges: "start:step:end"
z = [1;2;3;4] # Column Vectors
test = x === z' # Adjoint transpose

x = zeros(10,10,10) # Zero matrix
rowdata = x[1,:,:] # Subsection indexing

data = rand(50, 50)
firstrow = data[1,:]
firstrow = firstrow'
firstcol = data[:,1]

inner_prod = firstrow * firstcol # Vector/Matrix multiplication
outer_prod = firstcol * firstrow

firstrow_square = firstrow .* firstrow

zero_arr = floor.(firstrow) # Dot operator: performs a function elementwise on any AbstractArray

here = pwd() # (p)rint (w)orking (d)irectory
cd(here) #(c)hange (d)irectory
elsewhere = readdir() # List files in current directory

# Delimited file operations:
# randData=readdlm("Random.txt")
# writedlm(randData,"Random.csv")

# CSV file operations:
# data = CSV.read("")
# names(data)
# data[:someheader]

### Problems ###

#=
Example code
Generates a random array and uses it to simulate coin flips
=#

println("Heads percentage is: ",floor(mean(rand(100).>0.5)*10000)/100,"%")

#=
Exercise 1

Creates a file 'squares.txt' consisting of the first 5 square numbers
=#

writedlm("output/squares.txt", [i^2 for i=1:5])

#=
Exercise 2

Write a script which creates a new file called large_cities.txt.
The file should contain one line for each of the cities which have a
population larger than 10,000,000., formatted as follows:
    Buenos Aires, Argentina: population 11862073
    Sao Paulo, Brazil: population 14433147.5
    ...
=#

data = CSV.read("data/simplemaps-worldcities-basic.csv")
writedlm("output/large_cities.txt", map(row -> row.city * ", " * row.country * ": population " * @sprintf("%.1f", row.pop) , eachrow(filter(row -> row.pop > 10000000, data))))

#=
Exercise 3

Read data.txt given in the Public Folder and plot the results.
What do you see?
=#

plot(readdlm("data/data.txt")) # Hidden batman function!

#=
Exercise 4
	Plot a histogram of the longitudes of the world's cities. What is the mean and median longitude?
=#

histogram(data.lng)
;
