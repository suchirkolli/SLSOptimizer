"""
Create district reference data for Andhra Pradesh Smart Load Shedding Optimizer
"""

import pandas as pd
import os
from pathlib import Path

# Get absolute path to project root (assuming you're in scripts folder)
project_dir = Path(os.path.abspath(os.path.dirname(__file__))).parent
data_geo_dir = project_dir / 'data' / 'geographic'
os.makedirs(data_geo_dir, exist_ok=True)

# District data for Andhra Pradesh
districts_data = {
    'district_id': [f'AP-{i:02d}' for i in range(1, 14)],
    'district_name': [
        'Anantapur', 'Chittoor', 'East Godavari', 'Guntur', 'Kadapa',
        'Krishna', 'Kurnool', 'Prakasam', 'Nellore', 'Srikakulam',
        'Visakhapatnam', 'Vizianagaram', 'West Godavari'
    ],
    'population': [
        4500000, 4570000, 5380000, 5280000, 3140000,
        4810000, 4560000, 3710000, 3240000, 2850000,
        4720000, 2420000, 4050000
    ],
    'area_km2': [
        19130, 15152, 10807, 11391, 15359,
        8727, 17658, 17626, 13076, 5837,
        11161, 6539, 7742
    ]
}

# Create DataFrame
district_df = pd.DataFrame(districts_data)

# Calculate derived metrics
district_df['pop_density'] = district_df['population'] / district_df['area_km2']
district_df['pop_proportion'] = district_df['population'] / district_df['population'].sum()

# Calculate total population
total_population = district_df['population'].sum()
print(f"Total Andhra Pradesh population: {total_population:,}")

# Print district information with population density
print("\nDistrict Population Density:")
for _, district in district_df.sort_values('pop_density', ascending=False).iterrows():
    print(f"{district['district_name']}: {district['pop_density']:.1f} people/km²")

# Save to CSV using absolute path
output_path = data_geo_dir / 'ap_districts_reference.csv'
district_df.to_csv(output_path, index=False)
print(f"\nDistrict reference table saved to {output_path}")

# Create a geojson-like dictionary (simplified for demonstration)
# In a real project, you would use actual GeoJSON with proper boundaries
district_geo = {
    "type": "FeatureCollection",
    "features": []
}

for _, district in district_df.iterrows():
    # In a real project, you'd include actual polygon coordinates
    feature = {
        "type": "Feature",
        "properties": {
            "district_id": district["district_id"],
            "district_name": district["district_name"],
            "population": int(district["population"]),
            "pop_density": float(district["pop_density"])
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[]]  # Placeholder for actual coordinates
        }
    }
    district_geo["features"].append(feature)

# In a real project, you would save this as GeoJSON
# geo_path = data_geo_dir / 'ap_districts_geo.json' 
# with open(geo_path, 'w') as f:
#     json.dump(district_geo, f)

print("District reference processing complete")