import streamlit as st
import pickle
import numpy as np

# Load the model
rf = pickle.load(open('stacked_model.pkl', 'rb'))
cars = pickle.load(open('cars.pkl', 'rb'))

# Load the mappings
label_mappings = pickle.load(open('label_mappings.pkl', 'rb'))
frequency_mappings = pickle.load(open('frequency_mappings.pkl', 'rb'))

# Decode Label Encoding
for col in label_mappings:
    inv_map = {v: k for k, v in label_mappings[col].items()}
    cars[col] = cars[col].map(inv_map)

# Decode Frequency Encoding
for col in frequency_mappings:
    inv_map = {v: k for k, v in frequency_mappings[col].items()}
    cars[col] = cars[col].map(inv_map)

st.title("Car Price Predictor")

# Select inputs
Vehicle_make_display = st.selectbox('Make', sorted(cars['Vehicle_maker'].astype(str).unique()))
Vehicle_model_display = st.selectbox('Model', sorted(cars['Vehicle_line'].astype(str).unique()))
Location_display = st.selectbox('Location', sorted(cars['VRLOCATION'].astype(str).unique()))
VRMILEAGE = st.number_input('Mileage')
Vehicle_year = st.number_input('Vehicle Year')
Vehicle_cylinders = st.number_input('Cylinders')
Vehicle_doors = st.number_input('Doors')
Vehicle_engine = st.number_input('Engine')
Vehicle_condition_overall = st.number_input('Overall Condition')
Vehicle_condition_grade_1 = st.number_input('Condition Grade 1')
Vehicle_condition_grade_2 = st.number_input('Condition Grade 2')
VRSELLTYPE_display = st.selectbox('Sell Type', sorted(cars['VRSELLTYPE'].astype(str).unique()))
Vehicle_condition_crstatus_display = st.selectbox('Condition CR Status', sorted(cars['Vehicle_condition_crstatus'].astype(str).unique()))
Vehicle_condition_drivable_display = st.selectbox('Drivable', ["NaN", "Y", "N"])
Vehicle_airbag_display = st.selectbox('Airbag', ["NaN", "Y", "N"])
Misc_SalesChannel_display = st.selectbox('Sales Channel', sorted(cars['Misc_SalesChannel'].astype(str).unique()))
Vehicle_botcolor_display = st.selectbox('Bottom Color', sorted(cars['Vehicle_botcolor'].astype(str).unique()))
Vehicle_trantype_display = st.selectbox('Transmission Type', sorted(cars['Vehicle_trantype'].astype(str).unique()))
Vehicle_fuel_display = st.selectbox('Fuel Type', sorted(cars['Vehicle_fuel'].astype(str).unique()))
Vehicle_btext_display = st.selectbox('Body Text', sorted(cars['Vehicle_btext'].astype(str).unique()))
Vehicle_drive_display = st.selectbox('Drive Type', sorted(cars['Vehicle_drive'].astype(str).unique()))

# Map the selected decoded values back to the encoded values for prediction
def get_encoded_value(mapping, display_value):
    if display_value == "NaN":
        return np.nan
    if display_value in mapping:
        return mapping[display_value]
    else:
        st.error(f"Error: '{display_value}' not found in mapping.")
        return None

Vehicle_make = get_encoded_value(frequency_mappings.get('Vehicle_maker', {}), Vehicle_make_display)
Vehicle_model = get_encoded_value(frequency_mappings.get('Vehicle_line', {}), Vehicle_model_display)
Location = get_encoded_value(frequency_mappings.get('VRLOCATION', {}), Location_display)
VRSELLTYPE = get_encoded_value(label_mappings.get('VRSELLTYPE', {}), VRSELLTYPE_display)
Vehicle_condition_crstatus = get_encoded_value(label_mappings.get('Vehicle_condition_crstatus', {}), Vehicle_condition_crstatus_display)
Vehicle_condition_drivable = get_encoded_value(label_mappings.get('Vehicle_condition_drivable', {}), Vehicle_condition_drivable_display)
Vehicle_airbag = get_encoded_value(label_mappings.get('Vehicle_airbag', {}), Vehicle_airbag_display)
Misc_SalesChannel = get_encoded_value(label_mappings.get('Misc_SalesChannel', {}), Misc_SalesChannel_display)
Vehicle_botcolor = get_encoded_value(frequency_mappings.get('Vehicle_botcolor', {}), Vehicle_botcolor_display)
Vehicle_trantype = get_encoded_value(label_mappings.get('Vehicle_trantype', {}), Vehicle_trantype_display)
Vehicle_fuel = get_encoded_value(label_mappings.get('Vehicle_fuel', {}), Vehicle_fuel_display)
Vehicle_btext = get_encoded_value(frequency_mappings.get('Vehicle_btext', {}), Vehicle_btext_display)
Vehicle_drive = get_encoded_value(label_mappings.get('Vehicle_drive', {}), Vehicle_drive_display)

if st.button('Predict Price'):
    if None not in [Vehicle_make, Vehicle_model, Location, VRSELLTYPE,
                    Vehicle_condition_crstatus, Vehicle_condition_drivable, Misc_SalesChannel, Vehicle_botcolor,
                    Vehicle_trantype, Vehicle_fuel, Vehicle_btext,
                    Vehicle_drive, Vehicle_airbag]:
        # Prepare the input array
        query = np.array([VRMILEAGE, Vehicle_year, Vehicle_cylinders, Vehicle_doors, Vehicle_engine,
                          Vehicle_condition_overall, Vehicle_condition_grade_1, Vehicle_condition_grade_2, VRSELLTYPE,
                          Location, Vehicle_make, Vehicle_model,
                          Vehicle_condition_crstatus, Vehicle_condition_drivable, Misc_SalesChannel, Vehicle_botcolor,
                          Vehicle_trantype, Vehicle_fuel, Vehicle_btext,
                          Vehicle_drive, Vehicle_airbag])

        # Adjust for missing features
        query = np.pad(query, (0, 5), mode='constant', constant_values=np.nan)

        query = query.reshape(1, -1)
        prediction = rf.predict(query)
        st.title(f"The predicted price of this configuration is ${prediction[0]:.2f}")





# # Decode Label Encoding
# for col in label_mappings:
#     inv_map = {v: k for k, v in label_mappings[col].items()}
#     cars[col] = cars[col].map(inv_map)

# # Decode Frequency Encoding
# for col in frequency_mappings:
#     inv_map = {v: k for k, v in frequency_mappings[col].items()}
#     cars[col] = cars[col].map(inv_map)


    # encoded_inputs = []
    # for col, value in zip(['Vehicle_make', 'Vehicle_model', 'Location_LocationName', 'VRSELLTYPE', 'Vehicle_Mileage_Tier', 
    #                        'Vehicle_Pricing_Tier', 'Vehicle_condition_crstatus', 'Vehicle_condition_drivable', 
    #                        'Misc_SalesChannel', 'Vehicle_botcolor', 'Vehicle_intcolor', 'Vehicle_inttype', 
    #                        'Vehicle_topcnstr', 'Vehicle_trantype', 'Vehicle_fuel', 'Vehicle_btext', 
    #                        'Vehicle_bshort', 'Vehicle_drive'], 
    #                       [Vehicle_make, Vehicle_model, Location_LocationName, VRSELLTYPE, Vehicle_Mileage_Tier, 
    #                        Vehicle_Pricing_Tier, Vehicle_condition_crstatus, Vehicle_condition_drivable, 
    #                        Misc_SalesChannel, Vehicle_botcolor, Vehicle_intcolor, Vehicle_inttype, 
    #                        Vehicle_topcnstr, Vehicle_trantype, Vehicle_fuel, Vehicle_btext, 
    #                        Vehicle_bshort, Vehicle_drive]):
    #     if col in label_mappings:
    #         encoded_inputs.append(label_mappings[col].get(value, -1))
    #     elif col in frequency_mappings:
    #         encoded_inputs.append(frequency_mappings[col].get(value, -1))
    #     else:
    #         encoded_inputs.append(value)

    # query = np.array([VRMILEAGE, Vehicle_year, Vehicle_cylinders, Vehicle_doors, Vehicle_engine,
    #                 Vehicle_condition_overall, Vehicle_condition_grade_1, Vehicle_condition_grade_2] + encoded_inputs)


    # st.write("Selected Values:")
    # st.write(f"Make: {Vehicle_make}")
    # st.write(f"Model: {Vehicle_model}")
    # st.write(f"Location: {Location_LocationName}")
    # st.write(f"Mileage: {VRMILEAGE}")
    # st.write(f"Vehicle Year: {Vehicle_year}")
    # st.write(f"Cylinders: {Vehicle_cylinders}")
    # st.write(f"Doors: {Vehicle_doors}")
    # st.write(f"Engine: {Vehicle_engine}")
    # st.write(f"Overall Condition: {Vehicle_condition_overall}")
    # st.write(f"Condition Grade 1: {Vehicle_condition_grade_1}")
    # st.write(f"Condition Grade 2: {Vehicle_condition_grade_2}")
    # st.write(f"Sell Type: {VRSELLTYPE}")
    # st.write(f"Mileage Tier: {Vehicle_Mileage_Tier}")
    # st.write(f"Pricing Tier: {Vehicle_Pricing_Tier}")
    # st.write(f"Condition CR Status: {Vehicle_condition_crstatus}")
    # st.write(f"Drivable: {Vehicle_condition_drivable}")
    # st.write(f"Sales Channel: {Misc_SalesChannel}")
    # st.write(f"Bottom Color: {Vehicle_botcolor}")
    # st.write(f"Interior Color: {Vehicle_intcolor}")
    # st.write(f"Interior Type: {Vehicle_inttype}")
    # st.write(f"Top Construction: {Vehicle_topcnstr}")
    # st.write(f"Transmission Type: {Vehicle_trantype}")
    # st.write(f"Fuel Type: {Vehicle_fuel}")
    # st.write(f"Body Text: {Vehicle_btext}")
    # st.write(f"Body Short: {Vehicle_bshort}")
    # st.write(f"Drive Type: {Vehicle_drive}")

