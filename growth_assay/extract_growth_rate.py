import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

def LR(X,y):
    # Initialize the Linear Regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X, y)

    # Extract the intercept and coefficients
    intercept = model.intercept_
    coefficients = model.coef_

    return intercept, coefficients

filename="20241114_Microbe_Isolates_22-11 and 22-5_Batch_2"
nrows=89-37# the last row index - the first row index in the excel file
ncol_plate = 10 # rows in the 96-well plate
nrow_plate = 5 # columns in the 96-well plate
df = pd.read_excel(
    filename+".xlsx",
    sheet_name="Sheet2",
    usecols='A:KC',
    skiprows=36,
    nrows=nrows
    )

time = np.array(df.iloc[0,1:])/3600
data = np.array(df.iloc[2:,1:])
print(data.shape)
extracted_data={"Loc":[],"Max Growth Rate":[],"Max OD":[],}



fig1,axes1=plt.subplots(nrow_plate,ncol_plate,figsize=(ncol_plate,nrow_plate))
axes1=axes1.flat
fig2,axes2=plt.subplots(nrow_plate,ncol_plate,figsize=(ncol_plate,nrow_plate))
axes2=axes2.flat
for c in range(nrows-2):
    ax1=axes1[c]
    ax2=axes2[c]
    OD = data[c, :]
    logOD = np.log10(OD)
    GR=[]
    for t in range(len(OD)-12):
        y_fit = logOD[t:t+12]
        x_fit = time[t:t+12].reshape(-1, 1) # to reshape the data into a 2-d array (to suffice linear regression requirement)
        _, gr = LR(x_fit,y_fit)
        GR.append(gr)
    ax1.plot(time, OD, c="k", linewidth=1)
    ax2.plot(time[0:-12], GR, c="k", linewidth=1)
    max_GR=np.max(GR)
    max_OD=np.max(OD)
    Loc=str(df.iloc[2+c,0])
    ax1.text(x=0.1,y=0.7,s=Loc,transform=ax1.transAxes)
    ax2.text(x=0.9, y=0.7, s=Loc, ha="right", transform=ax2.transAxes)
    extracted_data["Loc"].append(Loc)
    extracted_data["Max Growth Rate"].append(max_GR)
    extracted_data["Max OD"].append(max_OD)
    ax1.set_xlim([0,24])
    ax1.set_ylim([0,1.2])
    ax1.set_yticks([0,0.5,1])
    ax2.set_xlim([0,24])
    ax2.set_ylim([0,0.5])
    ax2.set_yticks([0,0.2,0.4])
    if c == (nrow_plate-1)*ncol_plate:
        ax1.set_xlabel("time")
        ax1.set_ylabel("OD600")
        ax2.set_xlabel("time")
        ax2.set_ylabel("GR (hr$^{-1}$)")
    else:
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
extracted_data=pd.DataFrame(extracted_data)
extracted_data.to_csv(filename+"_analysis.csv")
fig1.tight_layout()
fig2.tight_layout()
plt.show()