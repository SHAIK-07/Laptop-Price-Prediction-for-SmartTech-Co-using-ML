import pandas as pd
import gradio as gr
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('final dataset.csv')

df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')

df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')

touchscreen_list = []

# Iterate over each value in the 'ScreenResolution' column
for resolution in df['ScreenResolution']:
    # Check if 'Touchscreen' is present in the resolution string
    if 'Touchscreen' in resolution:
        # Append 1 to the list if 'Touchscreen' is present
        touchscreen_list.append(1)
    else:
        # Append 0 to the list if 'Touchscreen' is not present
        touchscreen_list.append(0)

# Creating a new column 'Touchscreen' in the DataFrame and assign the list to it

df['Touchscreen'] = touchscreen_list



IPS_list = []

for resolution in df['ScreenResolution']:
    
    if 'IPS' in resolution:
        IPS_list.append(1)
    else:
        IPS_list.append(0)


df['Ips'] = IPS_list


cpu_names = []  # List to store processed CPU names


for cpu in df['Cpu']:
    x =cpu.split()[0:3]
    cpu_names.append(" ".join(x))

# Creating a new column 'Cpu Name' in the DataFrame and assign the processed CPU names
df['Cpu Name'] = cpu_names

def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

processor = []  # List to store processed CPU brands

# Iterate over each value in the 'Cpu Name' column
for cpu_name in df['Cpu Name']:
    processor.append(fetch_processor(cpu_name))

# Create a new column 'Cpu brand' in the DataFrame and assign the processed CPU brands
df['processor'] = processor



df.drop(columns=['Cpu','Cpu Name'],inplace=True)



df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n=1, expand=True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()
df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace('\D', '',regex=True)
df['second'] = df['second'].str.replace('\D', '',regex=True)

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['Memory','first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage','Hybrid','Flash_Storage'],inplace=True)

df['Gpu_brand'] = df['Gpu'].apply(lambda x:x.split()[0])

df.drop(columns=['Gpu'],inplace=True)

df = df[df['Gpu_brand'] != 'ARM']

def cat_os(text):
    if text == 'Windows 10' or text == 'Windows 7' or text == 'Windows 10 S':
        return 'Windows'
    elif text == 'macOS' or text == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

os = []  # List to store processed os names

# Iterate over each value in the 'OpSys' column
for os_name in df['OpSys']:
    os.append(cat_os(os_name))

# Create a new column 'os' in the DataFrame and assign the processed os names
df['os'] = os


df.drop(columns=['OpSys'],inplace=True)

df.drop(columns=['Inches'],inplace=True)


#Model Training and Application 

X = df.drop(columns=['Price'])
y = np.log(df['Price'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_features = ['Ram', 'Weight', 'HDD', 'SSD']
categorical_features = ['Company', 'TypeName', 'processor', 'Gpu_brand', 'os']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


model = RandomForestRegressor(random_state=42)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])


pipeline.fit(X_train, y_train)

def predict_price(Company, TypeName, Ram, Weight, Inches, X_res, Y_res, Touchscreen, Ips, processor, HDD, SSD, Gpu_brand, os):
    
    ppi = (X_res**2 + Y_res**2)**0.5 / Inches
     
    input_data = pd.DataFrame({
        'Company': [Company],
        'TypeName': [TypeName],
        'Ram': [Ram],
        'Weight': [Weight],
        'Touchscreen': [Touchscreen],
        'Ips': [Ips],
        'processor': [processor],
        'HDD': [HDD],
        'SSD': [SSD],
        'Gpu_brand': [Gpu_brand],
        'os': [os],
        'ppi': [ppi]  
    })

    
    predicted_log_price = pipeline.predict(input_data)[0]
    predicted_price = np.exp(predicted_log_price)
    return round(predicted_price)

company_list = df["Company"].unique().tolist()
type_name_list = df["TypeName"].unique().tolist()
Ram_list = sorted(df["Ram"].unique().tolist()) 
processor_list = df["processor"].unique().tolist()
HDD_list = sorted(df["HDD"].unique().tolist())
SSD_list = sorted(df["SSD"].unique().tolist()) 
gpu_brand_list = df["Gpu_brand"].unique().tolist()
os_list = df["os"].unique().tolist()


interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(choices=company_list, label="Company"),
        gr.Radio(choices=type_name_list, label="Type Name"),
        gr.Dropdown(choices=Ram_list, label="RAM (GB)"),
        gr.Number(label="Weight (kg)"),
        gr.Number(label="Screen Size (inches)"),
        gr.Number(label="X resolution"),
        gr.Number(label="Y resolution"),
        gr.Checkbox(label="Touchscreen"),
        gr.Checkbox(label="IPS Display"),
        gr.Radio(choices=processor_list, label="Processor"),
        gr.Dropdown(choices=HDD_list, label="HDD Storage (GB)"),
        gr.Dropdown(choices=SSD_list, label="SSD Storage (GB)"),
        gr.Radio(choices=gpu_brand_list, label="GPU Brand (if applicable)"),
        gr.Radio(choices=os_list, label="Operating System"),
    ],
    outputs=[gr.Textbox(label="Estimated Price")],
    title="Laptop Price Prediction App",
    description="Get an estimated price for your desired laptop based on its specifications.",
    allow_flagging=False  # Allow for feedback and model improvement
)


interface.launch(share=True)
