using CSV, DataFrames, HTTP, Statistics

function load_breast_cancer()
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv"
    
    println("Downloading dataset from URL...")
    
    # Always download fresh from URL
    resp = HTTP.get(url)
    
    # Read directly from the response
    df = CSV.read(IOBuffer(resp.body), DataFrame)
    
    println("Dataset shape: $(size(df))")
    
    # Remove ID column
    select!(df, Not(:Id))
    
    # Extract target variable (Class: 0=benign, 1=malignant)
    y = Vector{Int}(df.Class)
    
    # Remove Class from features
    select!(df, Not(:Class))
    
    # Handle Bare.nuclei column which has "NA" strings
    # Convert to numeric, replacing "NA" with missing
    bare_nuclei = df[!, Symbol("Bare.nuclei")]
    
    # Convert strings to numbers, "NA" becomes missing
    numeric_vals = Vector{Union{Missing, Float64}}(undef, length(bare_nuclei))
    for i in 1:length(bare_nuclei)
        val = bare_nuclei[i]
        if val == "NA" || ismissing(val)
            numeric_vals[i] = missing
        else
            numeric_vals[i] = parse(Float64, String(val))
        end
    end
    
    # Calculate median of non-missing values
    non_missing = skipmissing(numeric_vals)
    median_val = median(collect(non_missing))
    
    # Replace missing with median and convert to Float64
    df[!, Symbol("Bare.nuclei")] = [ismissing(x) ? median_val : Float64(x) for x in numeric_vals]
    
    # Now all columns should be numeric - convert to Float64 matrix
    X = Matrix{Float64}(df)
    
    println("\n✓ Dataset loaded successfully!")
    println("  Samples: $(size(X, 1))")
    println("  Features: $(size(X, 2))")
    println("  Benign (0): $(sum(y .== 0))")
    println("  Malignant (1): $(sum(y .== 1))")
    
    return X, y
end
