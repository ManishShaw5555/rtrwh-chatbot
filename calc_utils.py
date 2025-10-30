# calc_utils.py
def harvest_water_cubic_meters(roof_area_m2: float, annual_rainfall_mm: float, runoff_coeff: float) -> float:
    rainfall_m = annual_rainfall_mm / 1000.0
    return roof_area_m2 * rainfall_m * runoff_coeff

def recommend_tank_size(roof_area_m2, annual_rainfall_mm, runoff_coeff, storage_months=2):
    annual_harvest = harvest_water_cubic_meters(roof_area_m2, annual_rainfall_mm, runoff_coeff)
    monthly = annual_harvest / 12.0
    return monthly * storage_months

# safe-check example
def plausible_check(value):
    if value < 0 or value > 1e6:
        raise ValueError("value implausible")
