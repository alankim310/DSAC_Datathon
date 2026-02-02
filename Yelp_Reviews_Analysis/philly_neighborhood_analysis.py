"""
Compute a Philadelphia burger "value score" by neighborhood.

Inputs:
- burgers.csv (expected to contain city, state, postal_code, and stars_x)

Outputs:
- philly-neighborhood-analysis.csv

What it does:
- Maps Philly ZIP codes to neighborhoods, aggregates ratings, merges rent estimates,
  and computes Value_Score = (Rating^2 / Rent) * 10.
"""

import pandas as pd
import numpy as np

def run_neighborhood_value_analysis():
    print("="*60)
    print("STARTING PHILADELPHIA BURGER VALUE ANALYSIS")
    print("="*60)

    # ---------------------------------------------------------
    # PART 1: LOAD AND CALCULATE NEIGHBORHOOD RATINGS
    # ---------------------------------------------------------
    print("\n[Step 1/3] Loading data and mapping to neighborhoods...")
    try:
        burgers_df = pd.read_csv('burgers.csv')
    except Exception as e:
        print(f"Failed to load burgers.csv: {e}")
        return

    # Filter for PA and Philadelphia
    philly_burgers = burgers_df[
        (burgers_df['state'] == 'PA') & 
        (burgers_df['city'].str.lower() == 'philadelphia')
    ].copy()
    
    if philly_burgers.empty:
        print("No Philadelphia data found in burgers.csv")
        return

    # Clean zip codes
    def clean_zip(x):
        try:
            val = float(x)
            return str(int(val))
        except (ValueError, TypeError):
            s = str(x).strip()
            if '-' in s:
                return s.split('-')[0]
            return s

    philly_burgers['zip_clean'] = philly_burgers['postal_code'].apply(clean_zip)

    # Neighborhood Mapping
    zip_to_neighborhood = {
        '19102': 'Center City (Commercial)',
        '19103': 'Center City West (Rittenhouse)',
        '19104': 'University City / West Philly',
        '19106': 'Old City / Society Hill',
        '19107': 'Center City East (Chinatown/Wash West)',
        '19111': 'Fox Chase / Burholme',
        '19112': 'Navy Yard',
        '19113': 'Airport',
        '19114': 'Northeast Philly (Torresdale)',
        '19115': 'Northeast Philly (Bustleton)',
        '19116': 'Northeast Philly (Somerton)',
        '19118': 'Chestnut Hill',
        '19119': 'Mt. Airy',
        '19120': 'Olney / Feltonville',
        '19121': 'North Philly (Brewerytown/Francisville)',
        '19122': 'North Philly (Kensington/Ludlow)',
        '19123': 'Northern Liberties / Callowhill',
        '19124': 'Frankford / Juniata',
        '19125': 'Fishtown / Kensington',
        '19126': 'West Oak Lane',
        '19127': 'Manayunk',
        '19128': 'Roxborough',
        '19129': 'East Falls',
        '19130': 'Fairmount / Art Museum',
        '19131': 'West Philly (Overbrook / Wynnfield)',
        '19132': 'North Philly (Strawberry Mansion)',
        '19133': 'North Philly (Fairhill)',
        '19134': 'Port Richmond / Kensington',
        '19135': 'Tacony / Wissinoming',
        '19136': 'Holmesburg',
        '19137': 'Bridesburg',
        '19138': 'West Oak Lane / Ogontz',
        '19139': 'West Philly (Cedar Park/Walnut Hill)',
        '19140': 'North Philly (Hunting Park)',
        '19141': 'Logan / Ogontz',
        '19142': 'Southwest Philly',
        '19143': 'West Philly (Cobbs Creek/Kingsessing)',
        '19144': 'Germantown',
        '19145': 'South Philly (Girard Estates/Packer Park)',
        '19146': 'South Philly (Grad Hospital/Point Breeze)',
        '19147': 'South Philly (Bella Vista/Queen Village)',
        '19148': 'South Philly (Pennsport/Whitman)',
        '19149': 'Northeast Philly (Mayfair)',
        '19150': 'Cedarbrook',
        '19151': 'Overbrook',
        '19152': 'Northeast Philly (Rhawnhurst)',
        '19153': 'Southwest Philly (Eastwick)',
        '19154': 'Northeast Philly (Parkwood)',
    }

    philly_burgers['Neighborhood'] = philly_burgers['zip_clean'].map(zip_to_neighborhood)
    philly_burgers['Neighborhood'] = philly_burgers['Neighborhood'].fillna('Unknown / Other Zip')

    # Aggregation
    df = philly_burgers.groupby('Neighborhood').agg(
        Average_Rating=('stars_x', 'mean'),
        Review_Count=('stars_x', 'count')
    ).reset_index()

    print(f"✓ Aggregated data for {len(df)} neighborhoods.")

    # ---------------------------------------------------------
    # PART 2: ADD LEASE PRICES
    # ---------------------------------------------------------
    print("\n[Step 2/3] Adding estimated market rent data (2024)...")
    
    # Average Retail Lease Prices (NNN $/sqft/year)
    rent_map = {
        'Center City West (Rittenhouse)': 115.0,  
        'Center City (Commercial)': 90.0,         
        'Center City East (Chinatown/Wash West)': 55.0,
        'University City / West Philly': 55.0,   
        'Fishtown / Kensington': 45.0,            
        'Northern Liberties / Callowhill': 45.0,
        'Old City / Society Hill': 40.0,
        'Manayunk': 32.5,                        
        'Chestnut Hill': 40.0,                   
        'Fairmount / Art Museum': 35.0,          
        'South Philly (Bella Vista/Queen Village)': 38.0, 
        'South Philly (Grad Hospital/Point Breeze)': 30.0,
        'South Philly (Pennsport/Whitman)': 25.0,
        'South Philly (Girard Estates/Packer Park)': 25.0,
        'Northeast Philly (Torresdale)': 23.0,
        'Northeast Philly (Bustleton)': 23.0,
        'Northeast Philly (Somerton)': 23.0,
        'Northeast Philly (Mayfair)': 23.0,
        'Northeast Philly (Rhawnhurst)': 23.0,
        'Northeast Philly (Parkwood)': 23.0,
        'Roxborough': 25.0,
        'East Falls': 25.0,
        'Navy Yard': 40.0,                       
        'West Philly (Overbrook / Wynnfield)': 20.0,
        'West Philly (Cedar Park/Walnut Hill)': 25.0,
        'West Philly (Cobbs Creek/Kingsessing)': 18.0,
        'West Oak Lane': 18.0,
        'West Oak Lane / Ogontz': 18.0,
        'Germantown': 20.0,
        'Mt. Airy': 25.0,
        'North Philly (Brewerytown/Francisville)': 28.0, 
        'North Philly (Kensington/Ludlow)': 18.0,
        'North Philly (Strawberry Mansion)': 15.0,
        'North Philly (Fairhill)': 15.0,
        'North Philly (Hunting Park)': 15.0,
        'Port Richmond / Kensington': 22.0,
        'Bridesburg': 20.0,
        'Tacony / Wissinoming': 18.0,
        'Holmesburg': 18.0,
        'Frankford / Juniata': 18.0,
        'Fox Chase / Burholme': 20.0,
        'Olney / Feltonville': 18.0,
        'Logan / Ogontz': 18.0,
        'Cedarbrook': 18.0,
        'Overbrook': 20.0,
        'Southwest Philly': 18.0,
        'Southwest Philly (Eastwick)': 18.0,
        'Airport': 60.0, 
    }
    
    df['Avg_Rent_Per_SqFt'] = df['Neighborhood'].map(rent_map)
    
    # Fill missing with mean
    mean_rent = df['Avg_Rent_Per_SqFt'].mean()
    df['Avg_Rent_Per_SqFt'] = df['Avg_Rent_Per_SqFt'].fillna(mean_rent)
    print("✓ Rent data mapped.")

    # ---------------------------------------------------------
    # PART 3: CALCULATE VALUE SCORE
    # ---------------------------------------------------------
    print("\n[Step 3/3] Calculating Value Score = (Rating² / Rent) * 10...")

    # Value Metric formula
    df['Value_Score'] = ((df['Average_Rating'] ** 2) / df['Avg_Rent_Per_SqFt']) * 10
    
    # Sort and Format
    df = df.sort_values('Value_Score', ascending=False)
    
    # Create a pretty version for printing
    display_df = df.copy()
    display_df['Avg_Rent_Per_SqFt'] = display_df['Avg_Rent_Per_SqFt'].map('${:,.2f}'.format)
    display_df['Average_Rating'] = display_df['Average_Rating'].map('{:.2f}'.format)
    display_df['Value_Score'] = display_df['Value_Score'].map('{:.2f}'.format)

    print("\n" + "="*80)
    print("FINAL RESULTS: TOP NEIGHBORHOODS FOR VALUE")
    print("="*80)
    print(display_df[['Neighborhood', 'Average_Rating', 'Avg_Rent_Per_SqFt', 'Value_Score']].head(20).to_string(index=False))

    # Save final output
    output_file = 'philly-neighborhood-analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Full results saved to {output_file}")

if __name__ == "__main__":
    run_neighborhood_value_analysis()
