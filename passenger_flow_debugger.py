# Save this as: test_custom_passenger_flow.py
import pandas as pd
import sys
import os

def test_passenger_flow_file(filepath):
    """Test a custom passenger flow CSV file."""
    
    print(f"🔍 TESTING PASSENGER FLOW FILE: {filepath}")
    print("=" * 60)
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        print(f"✅ Successfully loaded {len(df)} records")
        
        # Check required columns
        required_cols = ['datetime', 'bus_id', 'line_id', 'stop_id', 'boarding', 'alighting', 'current_load', 'new_load']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        print(f"✅ All required columns present")
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        
        # Basic validation
        print("\n📊 BASIC STATISTICS:")
        print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Unique buses: {len(df['bus_id'].unique())}")
        print(f"Unique lines: {len(df['line_id'].unique())}")
        print(f"Unique stops: {len(df['stop_id'].unique())}")
        
        # Passenger conservation check
        total_boarding = df['boarding'].sum()
        total_alighting = df['alighting'].sum()
        difference = total_boarding - total_alighting
        conservation_ratio = total_alighting / total_boarding if total_boarding > 0 else 0
        
        print(f"\n🚌 PASSENGER CONSERVATION:")
        print(f"Total Boarding:  {total_boarding:,}")
        print(f"Total Alighting: {total_alighting:,}")
        print(f"Difference:      {difference:,}")
        print(f"Conservation:    {conservation_ratio:.4f}")
        
        if abs(difference) <= max(100, 0.01 * total_boarding):
            print("✅ CONSERVATION: PASS")
        else:
            print("❌ CONSERVATION: FAIL")
        
        # Check for data issues
        print(f"\n🔍 DATA QUALITY:")
        negative_boarding = (df['boarding'] < 0).sum()
        negative_alighting = (df['alighting'] < 0).sum()
        negative_load = (df['new_load'] < 0).sum()
        
        print(f"Negative boarding records: {negative_boarding}")
        print(f"Negative alighting records: {negative_alighting}")
        print(f"Negative load records: {negative_load}")
        
        if negative_boarding + negative_alighting + negative_load == 0:
            print("✅ DATA QUALITY: PASS")
        else:
            print("⚠️  DATA QUALITY: Issues found")
        
        # Overcrowding analysis
        if 'occupancy_rate' in df.columns:
            overcrowded = (df['occupancy_rate'] > 1.0).sum()
            severe_overcrowding = (df['occupancy_rate'] > 1.5).sum()
            print(f"\n🚌 CROWDING ANALYSIS:")
            print(f"Overcrowded records (>100%): {overcrowded} ({overcrowded/len(df)*100:.1f}%)")
            print(f"Severely overcrowded (>150%): {severe_overcrowding} ({severe_overcrowding/len(df)*100:.1f}%)")
        
        print(f"\n🎉 FILE VALIDATION COMPLETE!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing file: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_custom_passenger_flow.py <path_to_csv_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        sys.exit(1)
    
    test_passenger_flow_file(filepath)