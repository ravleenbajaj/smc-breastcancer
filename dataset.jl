using CSV, DataFrames, HTTP

function load_breast_cancer()
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv"
    localfile = "wdbc_data.csv"
    
    # Download if file doesn't exist
    if !isfile(localfile)
        println("Downloading dataset to $localfile ...")
        resp = HTTP.get(url)
        open(localfile, "w") do f
            write(f, String(resp.body))
        end
    end
    
    df = CSV.read(localfile, DataFrame)
    
    println("Original columns: ", names(df))
    println("First few rows:")
    println(first(df, 3))
    
    # Drop ID column if present (check actual column names)
    id_columns = [:Id, :SampleCodeNumber, :id, :ID, Symbol("Sample code number")]
    for col in id_columns
        if col in names(df)
            select!(df, Not(col))
            println("Dropped column: $col")
            break
        end
    end
    
    # Find the class/target column
    class_columns = [:Class, :class, :diagnosis, :Diagnosis, :target, :Target]
    class_col = nothing
    
    for col in class_columns
        if col in names(df)
            class_col = col
            println("Found class column: $col")
            break
        end
    end
    
    if isnothing(class_col)
        error("Could not find class/target column. Available columns: $(names(df))")
    end
    
    # Convert Class → 0/1
    if eltype(df[!, class_col]) <: Integer
        # If numeric, assume 2=benign, 4=malignant (common encoding)
        df[!, :Class] = map(x -> x == 4 ? 1 : 0, df[!, class_col])
    else
        # If string, check for malignant/benign
        df[!, :Class] = map(x -> lowercase(string(x)) == "malignant" ? 1 : 0, df[!, class_col])
    end
    
    # Remove original class column if it's not named :Class
    if class_col != :Class
        select!(df, Not(class_col))
    end
    
    # Extract features (all columns except Class)
    X = Matrix(select(df, Not(:Class)))
    y = Vector(df.Class)
    
    println("\nFinal dataset shape:")
    println("  Features (X): $(size(X))")
    println("  Target (y): $(length(y))")
    println("  Class distribution: Benign=$(sum(y.==0)), Malignant=$(sum(y.==1))")
    
    return X, y
end
