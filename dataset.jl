using CSV, DataFrames, HTTP, Statistics

function load_breast_cancer()
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv"
    
    println("Downloading dataset from URL...")
    
    # Always download fresh from URL
    resp = HTTP.get(url)
    
    # Read directly from the response
    df = CSV.read(IOBuffer(resp.body), DataFrame)
    
    println("Columns in dataset: ", names(df))
    println("Dataset shape: $(size(df))")
    println("First few rows:")
    println(first(df, 3))
    
    # Remove ID column
    select!(df, Not(:Id))
    
    # Extract target variable (Class: 0=benign, 1=malignant)
    y = Vector{Int}(df.Class)
    
    # Remove Class from features
    select!(df, Not(:Class))
    
    # Handle missing values (NA in Bare.nuclei column)
    # Replace missing/NA with column median
    for col in names(df)
        col_data = df[!, col]
        if eltype(col_data) >: Missing || any(ismissing, col_data)
            # Find median of non-missing values
            non_missing = skipmissing(col_data)
            if length(non_missing) > 0
                median_val = median(collect(non_missing))
                df[!, col] = coalesce.(col_data, median_val)
            end
        end
    end
    
    # Convert all features to Float64
    X = Matrix{Float64}(df)
    
    println("\n✓ Dataset loaded successfully from URL!")
    println("  Number of samples: $(size(X, 1))")
    println("  Number of features: $(size(X, 2))")
    println("  Benign (0): $(sum(y .== 0))")
    println("  Malignant (1): $(sum(y .== 1))")
    println("\nFeature names:")
    for (i, name) in enumerate(["Cl.thickness", "Cell.size", "Cell.shape", 
                                  "Marg.adhesion", "Epith.c.size", "Bare.nuclei",
                                  "Bl.cromatin", "Normal.nucleoli", "Mitoses"])
        println("  $i. $name")
    end
    
    return X, y
end
