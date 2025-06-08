import kagglehub

# Download latest version
path = kagglehub.dataset_download("erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database")

print("Path to dataset files:", path)