from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.copy()

    # Drop car name column if present
    if 'Car_Name' in df.columns:
        df.drop('Car_Name', axis=1, inplace=True)

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]

    return X, y
