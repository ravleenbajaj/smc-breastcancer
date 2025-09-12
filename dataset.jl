# dataset.jl
using CSV, DataFrames, HTTP

function load_breast_cancer()
    url = "https://raw.githubusercontent.com/plotly/datasets/master/breast-cancer-wisconsin.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Clean dataset: drop ID, convert target to 0/1
    select!(df, Not(:Id))
    df[!, :Class] .= map(x -> x == 4 ? 1 : 0, df[!, :Class])

    X = Matrix(select(df, Not(:Class)))
    y = Vector(df.Class)

    return X, y
end
