import pandas as pd

data1 =  pd.read_csv("../../abo-images-small/images/metadata/images.csv")
data2 =  pd.read_csv("../../abo-listings/listings/metadata/merged_images_listings.csv")

data1.to_csv("../images.csv", index=False)
data2.to_csv("../merged_images_listings.csv",index=False)
print("loaded and saved\n")

